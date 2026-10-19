// sysaudio: capture macOS system audio with a Core Audio Process Tap (macOS 14.2+)
// and stream it to stdout as 16 kHz mono float32 frames for the Python app.
//
// Usage:
//   sysaudio capture [--rate 16000]   stream frames until stdin closes or SIGINT/SIGTERM
//   sysaudio permissions              print microphone authorization status as JSON
//
// Frame format on stdout (little endian):
//   uint32 magic 'SAU1' | uint32 sampleCount | uint64 endTimeNanos (CLOCK_UPTIME_RAW,
//   same clock as Python's time.monotonic on macOS) | float32 samples[sampleCount]
// Diagnostics go to stderr as one JSON object per line.

import AVFoundation
import AudioToolbox
import CoreAudio
import Foundation

// MARK: - logging

func emit(_ fields: [String: Any]) {
    if let data = try? JSONSerialization.data(withJSONObject: fields),
       let line = String(data: data, encoding: .utf8) {
        FileHandle.standardError.write((line + "\n").data(using: .utf8)!)
    }
}

struct CoreAudioError: Error, CustomStringConvertible {
    let what: String
    let status: OSStatus
    var description: String { "\(what) failed (OSStatus \(status))" }
}

func check(_ status: OSStatus, _ what: String) throws {
    if status != noErr { throw CoreAudioError(what: what, status: status) }
}

// MARK: - Core Audio helpers

func globalAddress(_ selector: AudioObjectPropertySelector) -> AudioObjectPropertyAddress {
    AudioObjectPropertyAddress(mSelector: selector,
                               mScope: kAudioObjectPropertyScopeGlobal,
                               mElement: kAudioObjectPropertyElementMain)
}

func defaultOutputDevice() throws -> AudioObjectID {
    var address = globalAddress(kAudioHardwarePropertyDefaultSystemOutputDevice)
    var device = AudioObjectID(kAudioObjectUnknown)
    var size = UInt32(MemoryLayout<AudioObjectID>.size)
    try check(AudioObjectGetPropertyData(AudioObjectID(kAudioObjectSystemObject), &address, 0, nil,
                                         &size, &device), "get default output device")
    return device
}

func deviceUID(_ device: AudioObjectID) throws -> String {
    var address = globalAddress(kAudioDevicePropertyDeviceUID)
    var uid: Unmanaged<CFString>?
    var size = UInt32(MemoryLayout<Unmanaged<CFString>?>.size)
    try check(AudioObjectGetPropertyData(device, &address, 0, nil, &size, &uid), "get device UID")
    guard let value = uid?.takeRetainedValue() else {
        throw CoreAudioError(what: "device UID empty", status: -1)
    }
    return value as String
}

func tapFormat(_ tap: AudioObjectID) throws -> AudioStreamBasicDescription {
    var address = globalAddress(kAudioTapPropertyFormat)
    var asbd = AudioStreamBasicDescription()
    var size = UInt32(MemoryLayout<AudioStreamBasicDescription>.size)
    try check(AudioObjectGetPropertyData(tap, &address, 0, nil, &size, &asbd), "get tap format")
    return asbd
}

let timebase: mach_timebase_info_data_t = {
    var info = mach_timebase_info_data_t()
    mach_timebase_info(&info)
    return info
}()

func hostTimeToNanos(_ hostTime: UInt64) -> UInt64 {
    hostTime &* UInt64(timebase.numer) / UInt64(timebase.denom)
}

// MARK: - capture

final class SystemAudioCapture {
    private let outputRate: Double
    private var tapID = AudioObjectID(kAudioObjectUnknown)
    private var aggregateID = AudioObjectID(kAudioObjectUnknown)
    private var procID: AudioDeviceIOProcID?
    private let ioQueue = DispatchQueue(label: "sysaudio.io", qos: .userInteractive)
    private let outQueue = DispatchQueue(label: "sysaudio.out", qos: .userInitiated)
    private var converter: AVAudioConverter?
    private var inFormat: AVAudioFormat!
    private var outFormat: AVAudioFormat!
    private var framesWritten: UInt64 = 0

    init(outputRate: Double) { self.outputRate = outputRate }

    func start() throws {
        let description = CATapDescription(stereoGlobalTapButExcludeProcesses: [])
        description.uuid = UUID()
        description.name = "openinterview system tap"
        description.isPrivate = true
        description.muteBehavior = .unmuted
        try check(AudioHardwareCreateProcessTap(description, &tapID), "create process tap")

        let asbd = try tapFormat(tapID)
        let outputUID = try deviceUID(try defaultOutputDevice())
        let aggregate: [String: Any] = [
            kAudioAggregateDeviceNameKey: "openinterview tap device",
            kAudioAggregateDeviceUIDKey: UUID().uuidString,
            kAudioAggregateDeviceMainSubDeviceKey: outputUID,
            kAudioAggregateDeviceIsPrivateKey: true,
            kAudioAggregateDeviceIsStackedKey: false,
            kAudioAggregateDeviceTapAutoStartKey: true,
            kAudioAggregateDeviceSubDeviceListKey: [[kAudioSubDeviceUIDKey: outputUID]],
            kAudioAggregateDeviceTapListKey: [[
                kAudioSubTapDriftCompensationKey: true,
                kAudioSubTapUIDKey: description.uuid.uuidString,
            ]],
        ]
        try check(AudioHardwareCreateAggregateDevice(aggregate as CFDictionary, &aggregateID),
                  "create aggregate device")

        let channels = Int(asbd.mChannelsPerFrame)
        let nonInterleaved = asbd.mFormatFlags & kAudioFormatFlagIsNonInterleaved != 0
        let isFloat = asbd.mFormatFlags & kAudioFormatFlagIsFloat != 0 && asbd.mBitsPerChannel == 32
        guard isFloat else {
            throw CoreAudioError(what: "unsupported tap format (expected float32)", status: -1)
        }
        inFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: asbd.mSampleRate,
                                 channels: 1, interleaved: false)
        outFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: outputRate,
                                  channels: 1, interleaved: false)
        if asbd.mSampleRate != outputRate {
            converter = AVAudioConverter(from: inFormat, to: outFormat)
            converter?.sampleRateConverterQuality = AVAudioQuality.high.rawValue
        }
        emit(["event": "format", "sample_rate": asbd.mSampleRate, "channels": channels,
              "interleaved": !nonInterleaved, "output_rate": outputRate])

        try check(AudioDeviceCreateIOProcIDWithBlock(&procID, aggregateID, ioQueue) {
            [weak self] _, inputData, inputTime, _, _ in
            self?.handle(inputData, inputTime, channels: channels, nonInterleaved: nonInterleaved)
        }, "create IO proc")
        try check(AudioDeviceStart(aggregateID, procID), "start device")
        emit(["event": "started"])
    }

    // Runs on the real-time IO queue: downmix into a fresh buffer, then hand off.
    private func handle(_ inputData: UnsafePointer<AudioBufferList>,
                        _ inputTime: UnsafePointer<AudioTimeStamp>,
                        channels: Int, nonInterleaved: Bool) {
        let buffers = UnsafeMutableAudioBufferListPointer(UnsafeMutablePointer(mutating: inputData))
        guard let first = buffers.first, let firstData = first.mData else { return }
        let frames = nonInterleaved
            ? Int(first.mDataByteSize) / MemoryLayout<Float>.size
            : Int(first.mDataByteSize) / (MemoryLayout<Float>.size * max(channels, 1))
        guard frames > 0 else { return }
        guard let mono = AVAudioPCMBuffer(pcmFormat: inFormat, frameCapacity: AVAudioFrameCount(frames))
        else { return }
        mono.frameLength = AVAudioFrameCount(frames)
        let dst = mono.floatChannelData![0]
        let scale = 1.0 / Float(max(channels, 1))
        if nonInterleaved {
            dst.update(repeating: 0, count: frames)
            for b in buffers {
                guard let p = b.mData?.assumingMemoryBound(to: Float.self) else { continue }
                for i in 0..<frames { dst[i] += p[i] * scale }
            }
        } else {
            let p = firstData.assumingMemoryBound(to: Float.self)
            for i in 0..<frames {
                var sum: Float = 0
                for c in 0..<channels { sum += p[i * channels + c] }
                dst[i] = sum * scale
            }
        }
        let ts = inputTime.pointee
        var endNanos = UInt64(clock_gettime_nsec_np(CLOCK_UPTIME_RAW))
        if ts.mFlags.contains(.hostTimeValid) {
            let duration = UInt64(Double(frames) / inFormat.sampleRate * 1e9)
            endNanos = hostTimeToNanos(ts.mHostTime) &+ duration
        }
        outQueue.async { [weak self] in self?.convertAndWrite(mono, endNanos: endNanos) }
    }

    private func convertAndWrite(_ input: AVAudioPCMBuffer, endNanos: UInt64) {
        var output = input
        if let converter {
            let capacity = AVAudioFrameCount(Double(input.frameLength) * outputRate / inFormat.sampleRate) + 64
            guard let out = AVAudioPCMBuffer(pcmFormat: outFormat, frameCapacity: capacity) else { return }
            var supplied = false
            var error: NSError?
            converter.convert(to: out, error: &error) { _, status in
                if supplied { status.pointee = .noDataNow; return nil }
                supplied = true
                status.pointee = .haveData
                return input
            }
            if let error { emit(["event": "error", "message": "convert: \(error)"]); return }
            output = out
        }
        let count = Int(output.frameLength)
        guard count > 0 else { return }
        var header = Data(capacity: 16)
        withUnsafeBytes(of: UInt32(0x3155_4153).littleEndian) { header.append(contentsOf: $0) }
        withUnsafeBytes(of: UInt32(count).littleEndian) { header.append(contentsOf: $0) }
        withUnsafeBytes(of: endNanos.littleEndian) { header.append(contentsOf: $0) }
        let payload = Data(bytes: output.floatChannelData![0], count: count * MemoryLayout<Float>.size)
        FileHandle.standardOutput.write(header + payload)
        framesWritten += UInt64(count)
    }

    func stop() {
        if let procID {
            AudioDeviceStop(aggregateID, procID)
            AudioDeviceDestroyIOProcID(aggregateID, procID)
        }
        if aggregateID != kAudioObjectUnknown { AudioHardwareDestroyAggregateDevice(aggregateID) }
        if tapID != kAudioObjectUnknown { AudioHardwareDestroyProcessTap(tapID) }
        emit(["event": "stopped", "frames": framesWritten])
    }
}

// MARK: - commands

func microphoneStatus() -> String {
    switch AVCaptureDevice.authorizationStatus(for: .audio) {
    case .authorized: return "authorized"
    case .denied: return "denied"
    case .restricted: return "restricted"
    case .notDetermined: return "not_determined"
    @unknown default: return "unknown"
    }
}

func runCapture(rate: Double) -> Int32 {
    let capture = SystemAudioCapture(outputRate: rate)
    do {
        try capture.start()
    } catch {
        emit(["event": "error", "message": "\(error)"])
        capture.stop()
        return 1
    }
    let finish: () -> Never = {
        capture.stop()
        exit(0)
    }
    signal(SIGINT, SIG_IGN)
    signal(SIGTERM, SIG_IGN)
    let sigint = DispatchSource.makeSignalSource(signal: SIGINT, queue: .main)
    let sigterm = DispatchSource.makeSignalSource(signal: SIGTERM, queue: .main)
    sigint.setEventHandler { finish() }
    sigterm.setEventHandler { finish() }
    sigint.resume()
    sigterm.resume()
    // Parent closing our stdin (or dying) ends the capture.
    FileHandle.standardInput.readabilityHandler = { handle in
        if handle.availableData.isEmpty { DispatchQueue.main.async { finish() } }
    }
    dispatchMain()
}

// MARK: - TCC responsibility

// macOS attributes privacy permissions to the "responsible" process: normally the app that
// launched us (Terminal, an IDE, ...), which may have no audio-capture usage string and then
// gets silently denied. Re-exec'ing ourselves with responsibility disclaimed makes TCC treat
// this binary as its own client, so it prompts using the usage strings in our embedded
// Info.plist and the grant does not depend on which app launched us. This is the same
// (private, long-standing) libquarantine call that terminals and IDEs use for child processes.
@_silgen_name("responsibility_spawnattrs_setdisclaim")
func responsibility_spawnattrs_setdisclaim(_ attrs: UnsafeMutablePointer<posix_spawnattr_t?>,
                                           _ disclaim: Int32) -> Int32

func reexecAsResponsibleProcess() {
    let marker = "SYSAUDIO_RESPONSIBLE"
    guard getenv(marker) == nil else { return }
    setenv(marker, "1", 1)
    var attr: posix_spawnattr_t?
    posix_spawnattr_init(&attr)
    posix_spawnattr_setflags(&attr, Int16(POSIX_SPAWN_SETEXEC))  // replace this process image
    _ = responsibility_spawnattrs_setdisclaim(&attr, 1)
    let path = Bundle.main.executablePath ?? CommandLine.arguments[0]
    var argv: [UnsafeMutablePointer<CChar>?] = CommandLine.arguments.map { strdup($0) } + [nil]
    var pid: pid_t = 0
    let rc = posix_spawn(&pid, path, nil, &attr, &argv, environ)
    // Only reached if the exec failed; carry on as a normal child of the launcher.
    emit(["event": "warning", "message": "could not disclaim responsibility (errno \(rc))"])
}

let args = CommandLine.arguments
let command = args.count > 1 ? args[1] : "capture"
if !args.contains("--inherit-permissions") { reexecAsResponsibleProcess() }
switch command {
case "permissions":
    print("{\"microphone\": \"\(microphoneStatus())\", \"system_audio\": \"unknown_until_capture\"}")
case "capture":
    var rate = 16000.0
    if let i = args.firstIndex(of: "--rate"), i + 1 < args.count, let r = Double(args[i + 1]) { rate = r }
    exit(runCapture(rate: rate))
default:
    FileHandle.standardError.write(
        "usage: sysaudio [capture [--rate HZ] | permissions] [--inherit-permissions]\n".data(using: .utf8)!)
    exit(2)
}
