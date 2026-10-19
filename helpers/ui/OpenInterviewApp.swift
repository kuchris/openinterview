// OpenInterview native window (SwiftUI). A thin client for the local Python backend:
// it talks to the same HTTP + WebSocket API as the web dashboard (src/openinterview/web).
//
//   openinterview-ui --port 8765
//
// Layout (Views.swift): a native window toolbar (recording indicator, Answers toggle,
// settings, Start/Stop, transcript); a sidebar listing questions; the selected question's
// answer filling the main area; a two-line live caption bar; the full transcript in an
// inspector panel. A normal, visible window.

import AppKit
import SwiftUI

let port: Int = {
    let args = CommandLine.arguments
    if let i = args.firstIndex(of: "--port"), i + 1 < args.count, let p = Int(args[i + 1]) { return p }
    return 8765
}()
let baseURL = URL(string: "http://127.0.0.1:\(port)")!

// MARK: - model

struct Line: Identifiable, Equatable {
    let id: String
    let source: String
    let text: String
    let start: Double
    var stable: String = ""
    var unstable: String = ""
}

struct Card: Identifiable, Equatable {
    let id: Int
    var question = ""
    var source = ""
    var status = ""
    var error = ""
    var shortAnswer = ""
    var keyPoints: [String] = []
    var example = ""
    var model = ""
    var seconds = 0.0
    var spoken: [String] = []
    var covered: [String] = []
    var missed: [String] = []
    var tip = ""
    var feedbackStatus = ""

    init(_ d: [String: Any]) {
        id = d["id"] as? Int ?? 0
        question = d["question"] as? String ?? ""
        source = d["source"] as? String ?? ""
        status = d["status"] as? String ?? ""
        error = d["error"] as? String ?? ""
        spoken = d["spoken"] as? [String] ?? []
        feedbackStatus = d["feedback_status"] as? String ?? ""
        if let a = d["answer"] as? [String: Any] {
            shortAnswer = a["short_answer"] as? String ?? ""
            keyPoints = a["key_points"] as? [String] ?? []
            example = a["python_example"] as? String ?? ""
            model = a["model"] as? String ?? ""
            seconds = a["seconds"] as? Double ?? 0
        }
        if let f = d["feedback"] as? [String: Any] {
            covered = f["covered"] as? [String] ?? []
            missed = f["missed"] as? [String] ?? []
            tip = f["tip"] as? String ?? ""
        }
    }
}

func speaker(_ source: String) -> String {
    switch source {
    case "mic": return "Me"
    case "system": return "Them"
    case "typed": return "Typed"
    default: return source
    }
}

@MainActor
final class AppModel: ObservableObject {
    @Published var state = "idle"
    @Published var error = ""
    @Published var connected = false
    @Published var models: [String] = []
    @Published var devices: [String] = []
    @Published var source = "mic"
    @Published var model = "small.en"
    @Published var device = ""
    @Published var lines: [Line] = []
    @Published var partials: [String: Line] = [:]
    @Published var sourceStates: [String: String] = [:]
    @Published var finalLatency = "–"
    @Published var memory = "–"
    @Published var assistantOn = false
    @Published var assistantHealthy = false
    @Published var assistantHealth = ""
    @Published var listenTo: [String] = []
    @Published var profileLoaded = false
    @Published var profileWords = 0
    @Published var profilePath = ""
    @Published var profileMessage = ""
    @Published var profileToAdd: [String] = []
    @Published var profileBusy = false
    @Published var cards: [Card] = []
    @Published var selectedCard: Int?
    private var socket: URLSessionWebSocketTask?
    private var booted = false

    var running: Bool { state == "running" || state == "loading" }

    func boot() {
        guard !booted else { return }
        booted = true
        Task {
            await loadOptions()
            connect()
            if UserDefaults.standard.bool(forKey: "assistantOn") { setAssistant(true) }
        }
    }

    // -- HTTP ----------------------------------------------------------------------------

    @discardableResult
    func request(_ method: String, _ path: String, _ body: [String: Any]? = nil,
                 timeout: TimeInterval = 30) async -> [String: Any]? {
        var req = URLRequest(url: baseURL.appendingPathComponent(path))
        req.httpMethod = method
        req.timeoutInterval = timeout
        if let body {
            req.setValue("application/json", forHTTPHeaderField: "Content-Type")
            req.httpBody = try? JSONSerialization.data(withJSONObject: body)
        }
        guard let (data, resp) = try? await URLSession.shared.data(for: req) else {
            error = "Cannot reach the OpenInterview backend on port \(port)."
            return nil
        }
        let obj = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
        if let http = resp as? HTTPURLResponse, http.statusCode >= 400 {
            error = obj?["detail"] as? String ?? "Request failed (\(http.statusCode))"
            return nil
        }
        return obj
    }

    func loadOptions() async {
        guard let s = await request("GET", "api/state"),
              let o = s["options"] as? [String: Any] else { return }
        models = o["models"] as? [String] ?? []
        devices = (o["devices"] as? [[String: Any]] ?? []).compactMap { $0["name"] as? String }
        if let d = o["defaults"] as? [String: Any] {
            model = UserDefaults.standard.string(forKey: "model") ?? d["model"] as? String ?? model
            source = UserDefaults.standard.string(forKey: "source") ?? d["source"] as? String ?? source
            device = UserDefaults.standard.string(forKey: "device") ?? (d["mic_device"] as? String ?? "")
        }
        if !models.contains(model), let first = models.first { model = first }
        if !device.isEmpty && !devices.contains(device) { device = "" }
    }

    func toggleRecording() {
        error = ""
        Task {
            if running {
                await request("POST", "api/stop")
            } else {
                for (k, v) in [("model", model), ("source", source), ("device", device)] {
                    UserDefaults.standard.set(v, forKey: k)
                }
                await request("POST", "api/start", ["source": source, "model": model,
                                                     "mic_device": device.isEmpty ? NSNull() : device])
            }
        }
    }

    func setAssistant(_ on: Bool) {
        UserDefaults.standard.set(on, forKey: "assistantOn")
        Task { await request("POST", "api/assistant", ["enabled": on, "mode": "immediate"]) }
    }

    func ask(_ question: String) {
        let q = question.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !q.isEmpty else { return }
        Task { await request("POST", "api/assistant/ask", ["question": q]) }
    }

    func grade(_ id: Int) {
        Task { await request("POST", "api/assistant/\(id)/grade") }
    }

    // -- WebSocket -----------------------------------------------------------------------

    func connect() {
        let ws = URLSession.shared.webSocketTask(with: URL(string: "ws://127.0.0.1:\(port)/ws")!)
        socket = ws
        ws.resume()
        receive(ws)
    }

    private func receive(_ ws: URLSessionWebSocketTask) {
        ws.receive { [weak self] result in
            Task { @MainActor in
                guard let self else { return }
                switch result {
                case .success(let message):
                    self.connected = true
                    if case .string(let text) = message,
                       let data = text.data(using: .utf8),
                       let msg = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] {
                        self.handle(msg)
                    }
                    self.receive(ws)
                case .failure:
                    self.connected = false
                    try? await Task.sleep(nanoseconds: 1_000_000_000)
                    self.connect()
                }
            }
        }
    }

    private func line(_ e: [String: Any]) -> Line {
        let source = e["source"] as? String ?? ""
        return Line(id: "\(source):\(e["segment_id"] as? Int ?? 0)", source: source,
                    text: e["text"] as? String ?? "", start: e["start"] as? Double ?? 0,
                    stable: e["stable"] as? String ?? "", unstable: e["unstable"] as? String ?? "")
    }

    private func applyStatus(_ s: [String: Any]) {
        state = s["state"] as? String ?? state
        if let e = s["error"] as? String, !e.isEmpty { error = e }
    }

    private func applyAssistant(_ s: [String: Any]) {
        assistantOn = s["enabled"] as? Bool ?? false
        assistantHealthy = s["healthy"] as? Bool ?? false
        assistantHealth = s["health"] as? String ?? ""
        listenTo = s["listen_to"] as? [String] ?? []
        if let p = s["profile"] as? [String: Any] { applyProfile(p) }
    }

    private func applyProfile(_ p: [String: Any]) {
        profileLoaded = p["loaded"] as? Bool ?? false
        profileWords = p["words"] as? Int ?? 0
        profilePath = p["path"] as? String ?? ""
    }

    // -- personal profile ------------------------------------------------------------------

    func importCV() {
        let panel = NSOpenPanel()
        panel.title = "Import your CV"
        panel.allowedContentTypes = [.pdf, .rtf, .plainText, .text,
                                     .init(filenameExtension: "docx")!, .init(filenameExtension: "doc")!,
                                     .init(filenameExtension: "pages")!, .init(filenameExtension: "md")!]
        panel.allowsMultipleSelection = false
        guard panel.runModal() == .OK, let url = panel.url else { return }
        profileBusy = true
        profileToAdd = []
        profileMessage = "Analysing \(url.lastPathComponent) with AI…"
        Task {
            defer { profileBusy = false }
            if let info = await request("POST", "api/profile/import", ["path": url.path], timeout: 180) {
                applyProfile(info)
                profileToAdd = info["to_add"] as? [String] ?? []
                if (info["method"] as? String) == "ai" {
                    profileMessage = "Profile organised from \(url.lastPathComponent). Check it with Edit…"
                } else {
                    profileMessage = info["warning"] as? String ?? "Imported the raw CV text."
                }
            } else {
                profileMessage = error
            }
        }
    }

    func editProfile() {
        Task {
            if let info = await request("POST", "api/profile/edit") {
                applyProfile(info)
                NSWorkspace.shared.open(URL(fileURLWithPath: profilePath))
                profileMessage = "After saving your edits, click Reload."
            }
        }
    }

    func reloadProfile() {
        Task {
            if let info = await request("POST", "api/profile/reload") {
                applyProfile(info)
                profileMessage = profileLoaded ? "Profile reloaded." : "Profile is still empty."
            }
        }
    }

    private func handle(_ msg: [String: Any]) {
        switch msg["type"] as? String {
        case "snapshot":
            if let s = msg["status"] as? [String: Any] { applyStatus(s) }
            lines = (msg["lines"] as? [[String: Any]] ?? []).map(line)
            partials = [:]
        case "status":
            if (msg["state"] as? String) == "loading" {
                lines = []; partials = [:]; sourceStates = [:]; error = ""
            }
            applyStatus(msg)
        case "source":
            if let name = msg["source"] as? String { sourceStates[name] = msg["state"] as? String ?? "" }
        case "transcript":
            guard let e = msg["event"] as? [String: Any] else { return }
            let src = e["source"] as? String ?? ""
            switch e["kind"] as? String {
            case "partial":
                partials[src] = line(e)
            case "final":
                partials[src] = nil
                let l = line(e)
                if !l.text.isEmpty {
                    lines.append(l)
                    if lines.count > 300 { lines.removeFirst(lines.count - 300) }
                }
            case "retract":
                let id = line(e).id
                lines.removeAll { $0.id == id }
            default:
                break
            }
        case "metrics":
            let p = msg["pipeline"] as? [String: Any] ?? [:]
            if let f = p["final_latency_ms"] as? [String: Any], let v = f["p50"] as? Double {
                finalLatency = String(format: "%.2f s", v / 1000)
            }
            if let m = msg["memory"] as? [String: Any], let rss = m["rss_peak_mb"] as? Double {
                memory = "\(Int(rss + (m["mlx_peak_mb"] as? Double ?? 0))) MB"
            }
            for (name, st) in msg["sources"] as? [String: Any] ?? [:] {
                if let d = st as? [String: Any], let s = d["state"] as? String { sourceStates[name] = s }
            }
        case "assistant_snapshot":
            applyAssistant(msg)
            cards = (msg["cards"] as? [[String: Any]] ?? []).map(Card.init).sorted { $0.id > $1.id }
            if selectedCard == nil || !cards.contains(where: { $0.id == selectedCard }) {
                selectedCard = cards.first?.id
            }
        case "assistant_state":
            applyAssistant(msg)
        case "assistant_card":
            guard let d = msg["card"] as? [String: Any] else { return }
            let card = Card(d)
            if let i = cards.firstIndex(where: { $0.id == card.id }) {
                cards[i] = card
            } else {
                // Follow new questions unless the user is reading an older one.
                let following = selectedCard == nil || selectedCard == cards.first?.id
                cards.insert(card, at: 0)
                if following { selectedCard = card.id }
            }
        default:
            break
        }
    }
}

// Views live in Views.swift.

/// Capture this app's own window (no screen-recording permission is needed for your own
/// windows). CGWindowListCreateImage is looked up at runtime because newer SDKs hide it.
func captureOwnWindow(_ window: NSWindow) -> CGImage? {
    typealias Fn = @convention(c) (CGRect, UInt32, UInt32, UInt32) -> Unmanaged<CGImage>?
    guard let handle = dlopen("/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics", RTLD_NOW),
          let sym = dlsym(handle, "CGWindowListCreateImage") else { return nil }
    let fn = unsafeBitCast(sym, to: Fn.self)
    // .null rect, .optionIncludingWindow (8), .boundsIgnoreFraming (1)
    return fn(.null, 8, UInt32(window.windowNumber), 1)?.takeRetainedValue()
}

final class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.regular)
        NSApp.activate(ignoringOtherApps: true)
        // Development aid: render this window to a PNG (no screen-recording permission
        // needed, since it draws only its own view), e.g. for docs or UI checks.
        if let path = ProcessInfo.processInfo.environment["OPENINTERVIEW_UI_SNAPSHOT"] {
            let delay = Double(ProcessInfo.processInfo.environment["OPENINTERVIEW_UI_SNAPSHOT_DELAY"] ?? "") ?? 5
            DispatchQueue.main.asyncAfter(deadline: .now() + delay) {
                if let window = NSApp.windows.first(where: { $0.isVisible }),
                   let image = captureOwnWindow(window) {
                    let rep = NSBitmapImageRep(cgImage: image)
                    try? rep.representation(using: .png, properties: [:])?.write(to: URL(fileURLWithPath: path))
                }
                if ProcessInfo.processInfo.environment["OPENINTERVIEW_UI_SNAPSHOT_QUIT"] != nil {
                    NSApp.terminate(nil)
                }
            }
        }
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { true }
}

@main
struct OpenInterviewApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) var delegate
    @StateObject private var model = AppModel()

    var body: some Scene {
        WindowGroup("OpenInterview") {
            ContentView().environmentObject(model).onAppear { model.boot() }
        }
        .defaultSize(width: 1280, height: 580)
        .windowToolbarStyle(.unified)
    }
}
