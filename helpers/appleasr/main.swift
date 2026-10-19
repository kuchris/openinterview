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

/// Pause between words that splits a final into pieces. One final can span several
/// sentences, and Chinese / Japanese results often carry no punctuation between them.
let pieceGap = 0.6

/// Roughly how long a recognized run takes to say: ~0.4 s per Chinese / Japanese
/// character, ~0.08 s per letter otherwise.
func expectedDuration(_ text: String) -> Double {
    var cjk = 0, other = 0
    for scalar in text.unicodeScalars {
        switch scalar.value {
        case 0x3040...0x30FF, 0x3400...0x4DBF, 0x4E00...0x9FFF, 0xF900...0xFAFF: cjk += 1
        default: if CharacterSet.alphanumerics.contains(scalar) { other += 1 }
        }
    }
    return max(0.3, 0.4 * Double(cjk) + 0.08 * Double(other))
}

func emitResult(final: Bool, text: AttributedString, range: CMTimeRange) {
    // range covers the audio the recognizer has heard; words_end is where the last
    // recognized word actually ends (per-run audioTimeRange attributes).
    var wordsEnd = -1.0, wordsStart = -1.0
    var pieces: [[String: Any]] = []
    var current = "", start = -1.0, end = -1.0
    for run in text.runs {
        let piece = String(text[run.range].characters)
        if let tr = run.audioTimeRange {
            wordsEnd = max(wordsEnd, tr.end.seconds)
            if wordsStart < 0 { wordsStart = tr.start.seconds }
            // A pause shows up as a gap between words, or as a word far longer than it takes
            // to say (the recognizer gives the preceding silence to it: a Chinese character
            // spanning two seconds).
            let said = max(tr.start.seconds, tr.end.seconds - expectedDuration(piece))
            if final && end >= 0 && said - end >= pieceGap
                && !current.trimmingCharacters(in: .whitespaces).isEmpty {
                pieces.append(["text": current, "start": start, "end": end])
                current = ""
                start = -1
            }
            if start < 0 { start = said }
            end = max(end, tr.end.seconds)
        }
        current += piece
        // The recognizer gives the silence after a sentence to its punctuation ("。" can
        // span two seconds), so a long punctuation run is a pause too.
        if final, let tr = run.audioTimeRange,
           piece.trimmingCharacters(in: .whitespacesAndNewlines.union(.punctuationCharacters)).isEmpty,
           tr.end.seconds - tr.start.seconds >= pieceGap,
           !current.trimmingCharacters(in: .whitespacesAndNewlines.union(.punctuationCharacters)).isEmpty {
            pieces.append(["text": current, "start": start, "end": tr.start.seconds])
            current = ""
            start = -1
            end = -1
        }
    }
    var msg: [String: Any] = ["type": "result", "final": final, "text": String(text.characters),
                              "start": range.start.seconds, "end": range.end.seconds,
                              "words_start": wordsStart, "words_end": wordsEnd]
    // Finals always carry their pieces: their times are where speech was actually said,
    // so the caller can see pauses between finals too.
    if final {
        if !current.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && start >= 0 {
            pieces.append(["text": current, "start": start, "end": end])
        }
        msg["pieces"] = pieces
    }
    emit(msg)
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

/// An app may reserve only a few speech locales (5). Switching interview languages can hit
/// that limit ("Too many allocated locales"), so release another non-English reservation
/// to make room. The downloaded model stays on disk; reserving it again later is quick.
func makeRoomForLocale(_ locale: Locale) async {
    let wanted = locale.identifier(.bcp47)
    let reserved = await AssetInventory.reservedLocales
    guard !reserved.contains(where: { $0.identifier(.bcp47) == wanted }),
          reserved.count >= AssetInventory.maximumReservedLocales else { return }
    let others = reserved.filter { $0.identifier(.bcp47) != wanted }
    if let victim = others.first(where: { !$0.identifier(.bcp47).hasPrefix("en") }) ?? others.first {
        _ = await AssetInventory.release(reservedLocale: victim)
        emit(["type": "warning",
              "message": "released speech locale \(victim.identifier(.bcp47)) to make room for \(wanted)"])
    }
}

func ensureInstalled(_ transcriber: any SpeechModule, locale: Locale? = nil) async throws {
    if let locale { await makeRoomForLocale(locale) }
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
    try await ensureInstalled(module, locale: locale)

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
        try await ensureInstalled(makeTranscriber(locale: locale, fast: false), locale: locale)
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
