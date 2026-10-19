// appleasr: Apple's on-device streaming speech recognition (SpeechAnalyzer +
// SpeechTranscriber, macOS 26+) as a pipe filter for the Python app.
//
// Usage:
//   appleasr stream [--locale en-US] [--fast] [--dictation] [--context "term1, term2"]
//     default module: SpeechTranscriber (new long-form model); --dictation: DictationTranscriber
//   appleasr status [--locale en-US]          model availability / install state as JSON
//   appleasr install [--locale en-US]         download the on-device model if needed
//
// stdin:  frames (see helpers/common/common.swift). 'SAU1' = 16 kHz mono float32 audio whose
//         value is the absolute sample index of its first sample; 'SAUF' = finalize through
//         a sample index (sent by Python when its VAD sees end of speech).
// stdout: one JSON object per line:
//   {"type":"ready","format":...}
//   {"type":"result","final":bool,"text":...,"start":sec,"end":sec}
//            (start/end in the same timeline as the input sample indices / 16000)
//   {"type":"done","max_rss_mb":...}   {"type":"error","message":...}
// Unlike Whisper, this is genuine streaming: the recognizer consumes audio incrementally
// and revises a "volatile" tail until it finalizes it.

import AVFoundation
import CoreMedia
import Foundation
import Speech

let inputRate = 16000.0
let stdoutLock = NSLock()

func emit(_ fields: [String: Any]) {
    guard let data = try? JSONSerialization.data(withJSONObject: fields),
          let line = String(data: data, encoding: .utf8) else { return }
    stdoutLock.lock()
    FileHandle.standardOutput.write((line + "\n").data(using: .utf8)!)
    stdoutLock.unlock()
}

func option(_ name: String) -> String? {
    let args = CommandLine.arguments
    guard let i = args.firstIndex(of: name), i + 1 < args.count else { return nil }
    return args[i + 1]
}

func maxRSSMB() -> Double {
    var usage = rusage()
    getrusage(RUSAGE_SELF, &usage)
    return Double(usage.ru_maxrss) / 1_048_576  // bytes on macOS
}

func makeDictation(locale: Locale) -> DictationTranscriber {
    DictationTranscriber(locale: locale, contentHints: [], transcriptionOptions: [.punctuation],
                         reportingOptions: [.volatileResults, .frequentFinalization],
                         attributeOptions: [.audioTimeRange])
}

func emitResult(final: Bool, text: AttributedString, range: CMTimeRange) {
    // range covers the audio the recognizer has heard; words_end is where the last
    // recognized word actually ends (per-run audioTimeRange attributes).
    var wordsEnd = -1.0
    for run in text.runs {
        if let tr = run.audioTimeRange { wordsEnd = max(wordsEnd, tr.end.seconds) }
    }
    emit(["type": "result", "final": final, "text": String(text.characters),
          "start": range.start.seconds, "end": range.end.seconds, "words_end": wordsEnd])
}

func makeTranscriber(locale: Locale, fast: Bool) -> SpeechTranscriber {
    var reporting: Set<SpeechTranscriber.ReportingOption> = [.volatileResults]
    if fast { reporting.insert(.fastResults) }
    return SpeechTranscriber(locale: locale, transcriptionOptions: [],
                             reportingOptions: reporting, attributeOptions: [.audioTimeRange])
}

func resolveLocale() async throws -> Locale {
    let wanted = Locale(identifier: option("--locale") ?? "en-US")
    guard SpeechTranscriber.isAvailable else {
        throw NSError(domain: "appleasr", code: 1,
                      userInfo: [NSLocalizedDescriptionKey: "SpeechTranscriber is not available on this device"])
    }
    guard let locale = await SpeechTranscriber.supportedLocale(equivalentTo: wanted) else {
        throw NSError(domain: "appleasr", code: 2,
                      userInfo: [NSLocalizedDescriptionKey: "locale \(wanted.identifier) not supported"])
    }
    return locale
}

func statusString(_ s: AssetInventory.Status) -> String {
    switch s {
    case .unsupported: return "unsupported"
    case .supported: return "supported"
    case .downloading: return "downloading"
    case .installed: return "installed"
    @unknown default: return "unknown"
    }
}

func ensureInstalled(_ transcriber: any SpeechModule) async throws {
    let status = await AssetInventory.status(forModules: [transcriber])
    if status == .installed { return }
    if status == .unsupported {
        throw NSError(domain: "appleasr", code: 3,
                      userInfo: [NSLocalizedDescriptionKey: "speech model unsupported for this locale"])
    }
    if let request = try await AssetInventory.assetInstallationRequest(supporting: [transcriber]) {
        emit(["type": "installing"])
        try await request.downloadAndInstall()
        emit(["type": "installed"])
    }
}

// MARK: - stdin reader

/// Reads frames on a background thread and feeds the analyzer's input stream.
final class InputPump: @unchecked Sendable {
    let continuation: AsyncStream<AnalyzerInput>.Continuation
    let analyzer: SpeechAnalyzer
    let inFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: inputRate,
                                 channels: 1, interleaved: false)!
    let outFormat: AVAudioFormat
    let converter: AVAudioConverter?

    init(continuation: AsyncStream<AnalyzerInput>.Continuation, analyzer: SpeechAnalyzer,
         analyzerFormat: AVAudioFormat) {
        self.continuation = continuation
        self.analyzer = analyzer
        self.outFormat = analyzerFormat
        converter = analyzerFormat == inFormat ? nil : AVAudioConverter(from: inFormat, to: analyzerFormat)
    }

    func run(onEOF: @escaping @Sendable () -> Void) {
        let thread = Thread { [self] in
            let input = FileHandle.standardInput
            while true {
                guard let header = readExactly(input, 16) else { break }
                let magic = header.withUnsafeBytes { $0.load(fromByteOffset: 0, as: UInt32.self) }
                let count = Int(header.withUnsafeBytes { $0.load(fromByteOffset: 4, as: UInt32.self) })
                let value = header.withUnsafeBytes { $0.load(fromByteOffset: 8, as: UInt64.self) }
                if magic == frameMagicFinalize {
                    let through = CMTime(value: CMTimeValue(value), timescale: CMTimeScale(inputRate))
                    let analyzer = self.analyzer
                    Task { try? await analyzer.finalize(through: through) }
                    continue
                }
                guard magic == frameMagicAudio, let payload = readExactly(input, count * 4) else { break }
                feed(payload, count: count, startIndex: value)
            }
            continuation.finish()
            onEOF()
        }
        thread.qualityOfService = .userInitiated
        thread.start()
    }

    private func feed(_ payload: Data, count: Int, startIndex: UInt64) {
        guard count > 0,
              let buffer = AVAudioPCMBuffer(pcmFormat: inFormat, frameCapacity: AVAudioFrameCount(count))
        else { return }
        buffer.frameLength = AVAudioFrameCount(count)
        payload.withUnsafeBytes { raw in
            buffer.floatChannelData![0].update(from: raw.bindMemory(to: Float.self).baseAddress!,
                                               count: count)
        }
        var out = buffer
        if let converter {
            let ratio = outFormat.sampleRate / inFormat.sampleRate
            guard let converted = AVAudioPCMBuffer(pcmFormat: outFormat,
                                                   frameCapacity: AVAudioFrameCount(Double(count) * ratio) + 64)
            else { return }
            var supplied = false
            var error: NSError?
            converter.convert(to: converted, error: &error) { _, status in
                if supplied { status.pointee = .noDataNow; return nil }
                supplied = true
                status.pointee = .haveData
                return buffer
            }
            if let error { emit(["type": "error", "message": "convert: \(error)"]); return }
            out = converted
        }
        let start = CMTime(value: CMTimeValue(startIndex), timescale: CMTimeScale(inputRate))
        continuation.yield(AnalyzerInput(buffer: out, bufferStartTime: start))
    }
}

func readExactly(_ handle: FileHandle, _ n: Int) -> Data? {
    if n == 0 { return Data() }
    var data = Data(capacity: n)
    while data.count < n {
        guard let chunk = try? handle.read(upToCount: n - data.count), !chunk.isEmpty else { return nil }
        data.append(chunk)
    }
    return data
}

// MARK: - commands

func runStream() async throws {
    let locale = try await resolveLocale()
    let dictation = CommandLine.arguments.contains("--dictation")
    let module: any SpeechModule = dictation
        ? makeDictation(locale: locale)
        : makeTranscriber(locale: locale, fast: CommandLine.arguments.contains("--fast"))
    try await ensureInstalled(module)

    let analyzer = SpeechAnalyzer(modules: [module],
                                  options: .init(priority: .userInitiated, modelRetention: .processLifetime))
    guard let format = await SpeechAnalyzer.bestAvailableAudioFormat(compatibleWith: [module]) else {
        throw NSError(domain: "appleasr", code: 4,
                      userInfo: [NSLocalizedDescriptionKey: "no compatible audio format"])
    }
    try await analyzer.prepareToAnalyze(in: format)

    let (stream, continuation) = AsyncStream<AnalyzerInput>.makeStream()
    let results = Task {
        if let m = module as? DictationTranscriber {
            for try await r in m.results { emitResult(final: r.isFinal, text: r.text, range: r.range) }
        } else if let m = module as? SpeechTranscriber {
            for try await r in m.results { emitResult(final: r.isFinal, text: r.text, range: r.range) }
        }
    }
    try await analyzer.start(inputSequence: stream)
    if let terms = option("--context"), !terms.isEmpty {
        let context = AnalysisContext()
        context.contextualStrings[.general] = terms.split(separator: ",")
            .map { $0.trimmingCharacters(in: .whitespaces) }.filter { !$0.isEmpty }
        try await analyzer.setContext(context)
    }
    emit(["type": "ready", "locale": locale.identifier, "module": dictation ? "dictation" : "speech",
          "sample_rate": format.sampleRate,
          "channels": format.channelCount, "common_format": format.commonFormat.rawValue])

    let eof = AsyncStream<Void>.makeStream()
    InputPump(continuation: continuation, analyzer: analyzer, analyzerFormat: format)
        .run { eof.continuation.yield(); eof.continuation.finish() }
    for await _ in eof.stream { break }
    try await analyzer.finalizeAndFinishThroughEndOfInput()
    _ = try? await results.value
    emit(["type": "done", "max_rss_mb": maxRSSMB()])
}

let command = CommandLine.arguments.count > 1 ? CommandLine.arguments[1] : "stream"
if !CommandLine.arguments.contains("--inherit-permissions"),
   let rc = reexecAsResponsibleProcess(marker: "APPLEASR_RESPONSIBLE") {
    emit(["type": "warning", "message": "could not disclaim responsibility (errno \(rc))"])
}
signal(SIGPIPE, SIG_DFL)

do {
    switch command {
    case "status":
        let locale = try await resolveLocale()
        let status = await AssetInventory.status(forModules: [makeTranscriber(locale: locale, fast: false)])
        emit(["type": "status", "available": SpeechTranscriber.isAvailable,
              "locale": locale.identifier, "assets": statusString(status)])
    case "install":
        let locale = try await resolveLocale()
        try await ensureInstalled(makeTranscriber(locale: locale, fast: false))
        emit(["type": "installed", "locale": locale.identifier])
    case "stream":
        try await runStream()
    default:
        FileHandle.standardError.write("usage: appleasr [stream|status|install] [--locale en-US]\n"
            .data(using: .utf8)!)
        exit(2)
    }
} catch {
    emit(["type": "error", "message": error.localizedDescription])
    exit(1)
}
