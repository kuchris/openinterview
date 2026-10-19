// Saved sessions window: browse past sessions, replay their audio, review the questions,
// answers and transcript, and delete them.

import AVFoundation
import AppKit
import SwiftUI

struct SessionItem: Identifiable, Equatable {
    let id: String
    let title: String
    let created: String
    let duration: Int
    let company: String
    let role: String
    let questions: Int
    let hasAudio: Bool
    let size: Int

    init(_ d: [String: Any]) {
        id = d["id"] as? String ?? ""
        title = d["title"] as? String ?? ""
        created = d["created"] as? String ?? ""
        duration = d["duration_s"] as? Int ?? 0
        let t = d["target"] as? [String: Any] ?? [:]
        company = t["company"] as? String ?? ""
        role = t["role"] as? String ?? ""
        questions = d["questions"] as? Int ?? 0
        hasAudio = !((d["audio"] as? [String: Any]) ?? [:]).isEmpty
        size = d["size_bytes"] as? Int ?? 0
    }

    var date: String {
        let f = ISO8601DateFormatter()
        f.formatOptions = [.withFullDate, .withTime, .withColonSeparatorInTime]
        guard let d = f.date(from: created) ?? DateFormatter.iso.date(from: created) else { return created }
        return d.formatted(date: .abbreviated, time: .shortened)
    }
}

extension DateFormatter {
    static let iso: DateFormatter = {
        let f = DateFormatter()
        f.dateFormat = "yyyy-MM-dd'T'HH:mm:ss"
        return f
    }()
}

func durationText(_ s: Int) -> String {
    s >= 3600 ? String(format: "%d:%02d:%02d", s / 3600, s / 60 % 60, s % 60)
              : String(format: "%d:%02d", s / 60, s % 60)
}

struct SessionDetail {
    var title = ""
    var created = ""
    var duration = 0
    var target: [String: String] = [:]
    var cards: [Card] = []
    var transcript: [Line] = []
    var audio: [(source: String, path: String)] = []
    var folder = ""
}

@MainActor
final class HistoryModel: ObservableObject {
    @Published var sessions: [SessionItem] = []
    @Published var selected: String? { didSet { if selected != oldValue { loadDetail() } } }
    @Published var detail: SessionDetail?
    @Published var error = ""
    @Published var folder = ""
    private var observer: NSObjectProtocol?

    init() {
        observer = NotificationCenter.default.addObserver(forName: .sessionSaved, object: nil,
                                                          queue: .main) { [weak self] _ in
            Task { @MainActor in self?.refresh() }
        }
    }

    private func get(_ path: String, method: String = "GET") async -> [String: Any]? {
        var req = URLRequest(url: baseURL.appendingPathComponent(path))
        req.httpMethod = method
        guard let (data, resp) = try? await URLSession.shared.data(for: req) else {
            error = "Cannot reach the OpenInterview backend."
            return nil
        }
        let obj = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
        if let http = resp as? HTTPURLResponse, http.statusCode >= 400 {
            error = obj?["detail"] as? String ?? "Request failed (\(http.statusCode))"
            return nil
        }
        error = ""
        return obj
    }

    func refresh() {
        Task {
            guard let r = await get("api/sessions") else { return }
            folder = r["folder"] as? String ?? ""
            sessions = (r["sessions"] as? [[String: Any]] ?? []).map(SessionItem.init)
            if selected == nil || !sessions.contains(where: { $0.id == selected }) {
                selected = sessions.first?.id
            }
        }
    }

    func loadDetail() {
        guard let id = selected else { detail = nil; return }
        Task {
            guard let d = await get("api/sessions/\(id)") else { return }
            var s = SessionDetail()
            s.title = d["title"] as? String ?? ""
            s.created = d["created"] as? String ?? ""
            s.duration = d["duration_s"] as? Int ?? 0
            s.target = (d["target"] as? [String: String]) ?? [:]
            s.cards = (d["cards"] as? [[String: Any]] ?? []).map(Card.init)
            s.transcript = (d["transcript"] as? [[String: Any]] ?? []).map { e in
                Line(id: "\(e["source"] as? String ?? ""):\(e["segment_id"] as? Int ?? 0)",
                     source: e["source"] as? String ?? "", text: e["text"] as? String ?? "",
                     start: e["start"] as? Double ?? 0)
            }
            s.audio = ((d["audio_paths"] as? [String: String]) ?? [:]).sorted { $0.key < $1.key }
                .map { (source: $0.key, path: $0.value) }
            s.folder = d["folder"] as? String ?? ""
            if selected == id { detail = s }
        }
    }

    func delete(_ id: String) {
        Task {
            guard await get("api/sessions/\(id)", method: "DELETE") != nil else { return }
            if selected == id { selected = nil; detail = nil }
            refresh()
        }
    }
}

struct HistoryView: View {
    @EnvironmentObject var h: HistoryModel
    @State private var confirmDelete: String?

    var body: some View {
        NavigationSplitView {
            Group {
                if h.sessions.isEmpty {
                    VStack(spacing: 6) {
                        Image(systemName: "clock").font(.title).foregroundStyle(.tertiary)
                        Text("No saved sessions yet").foregroundStyle(.secondary)
                        Text("Each Start creates one.").font(.caption).foregroundStyle(.tertiary)
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                } else {
                    List(selection: $h.selected) {
                        ForEach(h.sessions) { s in
                            SessionRow(item: s).tag(s.id)
                                .contextMenu {
                                    Button("Delete…", role: .destructive) { confirmDelete = s.id }
                                }
                        }
                    }
                    .listStyle(.sidebar)
                }
            }
            .navigationSplitViewColumnWidth(min: 240, ideal: 290, max: 380)
        } detail: {
            if let d = h.detail {
                SessionDetailView(detail: d)
            } else {
                Text(h.error.isEmpty ? "Select a session" : h.error).foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        }
        .navigationTitle("Session history")
        .toolbar {
            ToolbarItemGroup(placement: .primaryAction) {
                Button { h.refresh() } label: { Label("Refresh", systemImage: "arrow.clockwise") }
                Button {
                    NSWorkspace.shared.open(URL(fileURLWithPath: h.detail?.folder ?? h.folder))
                } label: { Label("Show in Finder", systemImage: "folder") }
                .help("Open the session folder")
                Button(role: .destructive) { confirmDelete = h.selected } label: {
                    Label("Delete", systemImage: "trash")
                }
                .disabled(h.selected == nil)
                .help("Delete the selected session (text and audio)")
            }
        }
        .alert("Delete this session?", isPresented: Binding(get: { confirmDelete != nil },
                                                             set: { if !$0 { confirmDelete = nil } })) {
            Button("Delete", role: .destructive) {
                if let id = confirmDelete { h.delete(id) }
                confirmDelete = nil
            }
            Button("Cancel", role: .cancel) { confirmDelete = nil }
        } message: {
            Text("Its questions, answers, transcript and audio are removed from this Mac.")
        }
        .onAppear { h.refresh() }
        .frame(minWidth: 760, minHeight: 460)
    }
}

struct SessionRow: View {
    let item: SessionItem

    var body: some View {
        VStack(alignment: .leading, spacing: 3) {
            Text(item.title).font(.callout.weight(.semibold)).lineLimit(1)
            Text("\(item.date) · \(durationText(item.duration))")
                .font(.caption).foregroundStyle(.secondary)
            let target = [item.company, item.role].filter { !$0.isEmpty }.joined(separator: " · ")
            HStack(spacing: 8) {
                if !target.isEmpty { Text(target).lineLimit(1) }
                Label("\(item.questions)", systemImage: "questionmark.bubble")
                if item.hasAudio { Image(systemName: "waveform") }
            }
            .font(.caption2).foregroundStyle(.tertiary)
        }
        .padding(.vertical, 3)
    }
}

struct SessionDetailView: View {
    let detail: SessionDetail

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                VStack(alignment: .leading, spacing: 4) {
                    Text(detail.title).font(.system(size: 22, weight: .bold))
                    let target = [detail.target["company"], detail.target["role"]]
                        .compactMap { $0 }.filter { !$0.isEmpty }.joined(separator: " · ")
                    Text([target, durationText(detail.duration)].filter { !$0.isEmpty }
                        .joined(separator: " · ")).foregroundStyle(.secondary)
                }
                if !detail.audio.isEmpty {
                    SectionBox(title: "Audio", icon: "waveform") {
                        VStack(alignment: .leading, spacing: 8) {
                            ForEach(detail.audio, id: \.source) { a in
                                AudioRow(label: speaker(a.source), path: a.path)
                            }
                        }
                    }
                }
                if !detail.cards.isEmpty {
                    Text("Questions").font(.headline)
                    ForEach(detail.cards) { HistoryCard(card: $0) }
                }
                if !detail.transcript.isEmpty {
                    SectionBox(title: "Transcript", icon: "text.bubble") {
                        VStack(alignment: .leading, spacing: 5) {
                            ForEach(detail.transcript) { l in
                                HStack(alignment: .firstTextBaseline, spacing: 8) {
                                    Text(String(format: "%02d:%02d", Int(l.start) / 60, Int(l.start) % 60))
                                        .font(.caption.monospacedDigit()).foregroundStyle(.tertiary)
                                    Text(speaker(l.source)).font(.caption.weight(.bold))
                                        .foregroundStyle(l.source == "mic" ? Color.teal : Color.purple)
                                        .frame(width: 36, alignment: .leading)
                                    Text(l.text).textSelection(.enabled)
                                }
                            }
                        }
                    }
                }
            }
            .frame(maxWidth: 820, alignment: .leading)
            .padding(24)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }
}

struct HistoryCard: View {
    let card: Card

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(card.question).font(.title3.weight(.semibold)).textSelection(.enabled)
            if !card.error.isEmpty { Text(card.error).font(.callout).foregroundStyle(.orange) }
            if !card.shortAnswer.isEmpty { Text(card.shortAnswer).textSelection(.enabled) }
            ForEach(Array(card.keyPoints.enumerated()), id: \.offset) { i, p in
                Text("\(i + 1). \(p)").font(.callout).foregroundStyle(.secondary)
            }
            if !card.example.isEmpty { CodeBlock(code: card.example) }
            if !card.spoken.isEmpty {
                Text("You said: \(card.spoken.joined(separator: " "))")
                    .font(.callout).foregroundStyle(.secondary).textSelection(.enabled)
            }
            if !card.covered.isEmpty || !card.missed.isEmpty {
                FlowChips(items: card.covered, icon: "checkmark", color: .green)
                FlowChips(items: card.missed, icon: "xmark", color: .orange)
                if !card.tip.isEmpty { Label(card.tip, systemImage: "lightbulb").font(.callout) }
            }
        }
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(RoundedRectangle(cornerRadius: 12).fill(Color.primary.opacity(0.045)))
    }
}

@MainActor
final class Player: ObservableObject {
    @Published var playing = false
    @Published var position = 0.0
    @Published var duration = 0.0
    private var player: AVAudioPlayer?
    private var timer: Timer?

    func load(_ path: String) {
        player = try? AVAudioPlayer(contentsOf: URL(fileURLWithPath: path))
        player?.prepareToPlay()
        duration = player?.duration ?? 0
    }

    func toggle() {
        guard let p = player else { return }
        if p.isPlaying {
            p.pause()
            playing = false
            timer?.invalidate()
        } else {
            p.play()
            playing = true
            timer = Timer.scheduledTimer(withTimeInterval: 0.25, repeats: true) { [weak self] _ in
                Task { @MainActor in
                    guard let self, let p = self.player else { return }
                    self.position = p.currentTime
                    if !p.isPlaying { self.playing = false; self.timer?.invalidate() }
                }
            }
        }
    }

    func seek(_ t: Double) {
        player?.currentTime = t
        position = t
    }

    func stop() {
        player?.stop()
        timer?.invalidate()
    }
}

struct AudioRow: View {
    let label: String
    let path: String
    @StateObject private var player = Player()

    var body: some View {
        HStack(spacing: 10) {
            Button { player.toggle() } label: {
                Image(systemName: player.playing ? "pause.circle.fill" : "play.circle.fill").font(.title2)
            }
            .buttonStyle(.borderless)
            Text(label).font(.callout.weight(.semibold)).frame(width: 44, alignment: .leading)
            Slider(value: Binding(get: { player.position }, set: { player.seek($0) }),
                   in: 0...max(player.duration, 1))
            Text("\(durationText(Int(player.position))) / \(durationText(Int(player.duration)))")
                .font(.caption.monospacedDigit()).foregroundStyle(.secondary)
        }
        .onAppear { player.load(path) }
        .onDisappear { player.stop() }
        .id(path)
    }
}
