// Shared by the native helpers (sysaudio, appleasr).

import Foundation

// MARK: - TCC responsibility

// macOS attributes privacy permissions to the "responsible" process: normally the app that
// launched us (Terminal, an IDE, ...), which may have no usage string for the service and
// then gets silently denied. Re-exec'ing ourselves with responsibility disclaimed makes TCC
// treat this binary as its own client, so it prompts using the usage strings in our embedded
// Info.plist and the grant does not depend on which app launched us. This is the same
// (private, long-standing) libquarantine call that terminals and IDEs use for child processes.
@_silgen_name("responsibility_spawnattrs_setdisclaim")
func responsibility_spawnattrs_setdisclaim(_ attrs: UnsafeMutablePointer<posix_spawnattr_t?>,
                                           _ disclaim: Int32) -> Int32

/// Replaces the current process image with a responsibility-disclaimed copy of itself.
/// Returns nil if already disclaimed (marker set); otherwise only returns, with the errno,
/// if the exec failed.
func reexecAsResponsibleProcess(marker: String) -> Int32? {
    guard getenv(marker) == nil else { return nil }
    setenv(marker, "1", 1)
    var attr: posix_spawnattr_t?
    posix_spawnattr_init(&attr)
    posix_spawnattr_setflags(&attr, Int16(POSIX_SPAWN_SETEXEC))
    _ = responsibility_spawnattrs_setdisclaim(&attr, 1)
    let path = Bundle.main.executablePath ?? CommandLine.arguments[0]
    var argv: [UnsafeMutablePointer<CChar>?] = CommandLine.arguments.map { strdup($0) } + [nil]
    var pid: pid_t = 0
    return posix_spawn(&pid, path, nil, &attr, &argv, environ)
}

// MARK: - PCM frame protocol

// Little-endian frames: uint32 magic | uint32 sampleCount | uint64 value | float32[sampleCount]
//   'SAU1' audio:    sysaudio -> Python: value = capture end time (ns, CLOCK_UPTIME_RAW)
//                    Python -> appleasr: value = absolute sample index of samples[0]
//   'SAUF' finalize: Python -> appleasr: value = sample index to finalize through (count 0)
let frameMagicAudio: UInt32 = 0x3155_4153     // "SAU1"
let frameMagicFinalize: UInt32 = 0x4655_4153  // "SAUF"
