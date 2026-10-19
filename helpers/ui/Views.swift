// OpenInterview window layout.
//
//  ┌ toolbar: ● Recording · source/model subtitle ……… [Answers] [⚙︎] [Start/Stop] [Transcript] ┐
//  │ Questions (sidebar)  │  Selected question                                              │
//  │  ◌ How would you …   │  Short answer (large)                                           │
//  │  ✓ What is the …     │  ① ② ③ key points                                               │
//  │ [Ask a question…]    │  Python example [copy]   ·   Your answer + feedback              │
//  ├──────────────────────┴─────────────────────────────────────────────────────────────────┤
//  │ live caption: last line + partial (two lines)                       0.72 s · OpenCode │
//  └─────────────────────────────────────────────────────────────────────────────────────────┘
//  Full transcript: inspector panel on the right (toolbar button).

import AppKit
import SwiftUI

// MARK: - root

struct ContentView: View {
    @EnvironmentObject var m: AppModel
    @Environment(\.openWindow) private var openWindow
    @AppStorage("showTranscript") private var showTranscript = false
    @State private var showSettings = false

    var body: some View {
        NavigationSplitView {
            Sidebar()
                .navigationSplitViewColumnWidth(min: 210, ideal: 240, max: 340)
        } detail: {
            AnswerDetail()
                .safeAreaInset(edge: .bottom, spacing: 0) { CaptionBar() }
                .frame(minWidth: 440)
        }
        .inspector(isPresented: $showTranscript) {
            TranscriptInspector()
                .inspectorColumnWidth(min: 220, ideal: 250, max: 420)
        }
        .navigationTitle("OpenInterview")
        .navigationSubtitle(subtitle)
        .sheet(isPresented: $m.showNewSession) { NewSessionSheet() }
        .sheet(item: $m.noteCard) { card in AddNoteSheet(card: card) }
        .onAppear {
            // Development aid for UI checks: open a window or sheet at launch.
            switch ProcessInfo.processInfo.environment["OPENINTERVIEW_UI_OPEN"] {
            case "history": openWindow(id: "history")
            case "newsession": m.showNewSession = true
            default: break
            }
        }
        .toolbar {
            ToolbarItem(placement: .navigation) { RecordingBadge(state: m.state) }
            ToolbarItemGroup(placement: .primaryAction) {
                Toggle(isOn: Binding(get: { m.assistantOn }, set: { m.setAssistant($0) })) {
                    Label("Answers", systemImage: "sparkles")
                }
                .toggleStyle(.button)
                .help(m.assistantOn ? "Answers on: questions are answered automatically"
                                    : "Turn on automatic answers (needs your local OpenCode server)")
                Button { showSettings.toggle() } label: { Label("Settings", systemImage: "slider.horizontal.3") }
                    .help("Audio source, model and microphone")
                    .popover(isPresented: $showSettings, arrowEdge: .bottom) { SettingsPopover() }
                Button(action: m.toggleRecording) {
                    Label(m.running ? "Stop" : "Start", systemImage: m.running ? "stop.fill" : "record.circle")
                        .labelStyle(.titleAndIcon)
                }
                .buttonStyle(.borderedProminent)
                .tint(m.running ? .red : .accentColor)
                .keyboardShortcut("r", modifiers: .command)
                .disabled(m.state == "stopping")
                .help("Start or stop listening (⌘R)")
                Button { showTranscript.toggle() } label: { Label("Transcript", systemImage: "text.bubble") }
                    .help("Show the full transcript")
                Button { openWindow(id: "history") } label: {
                    Label("History", systemImage: "clock.arrow.circlepath")
                }
                .help("Saved sessions: questions, answers, transcript and audio")
            }
        }
        .frame(minWidth: 700, minHeight: 480)
    }

    var subtitle: String {
        let src = ["mic": "Microphone", "system": "System audio", "both": "Mic + system"][m.source] ?? m.source
        if m.running && !m.currentTitle.isEmpty {
            return "\(m.currentTitle) · \(src)\(m.currentSaving ? " · saving" : "")"
        }
        return "\(src) · \(m.model)" + (m.language == "en" ? "" : " · \(m.languageName)")
    }
}

// MARK: - toolbar pieces

struct RecordingBadge: View {
    let state: String
    @State private var pulse = false

    var body: some View {
        let on = state == "running"
        HStack(spacing: 6) {
            Circle().fill(on ? Color.red : Color.secondary.opacity(0.5))
                .frame(width: 8, height: 8)
                .opacity(on && pulse ? 0.3 : 1)
                .animation(on ? .easeInOut(duration: 0.8).repeatForever() : .default, value: pulse)
            Text(on ? "Recording" : state == "loading" ? "Starting…" : state == "stopping" ? "Stopping…" : "Idle")
                .font(.callout.weight(.semibold))
                .foregroundStyle(on ? Color.red : Color.secondary)
        }
        .padding(.horizontal, 9).padding(.vertical, 3)
        .background(Capsule().fill(on ? Color.red.opacity(0.13) : Color.secondary.opacity(0.1)))
        .onAppear { pulse = true }
        .accessibilityLabel(on ? "Recording" : "Not recording")
    }
}

struct SettingsPopover: View {
    @EnvironmentObject var m: AppModel

    var body: some View {
        Form {
            Section("Audio") {
                Picker("Listen to", selection: $m.source) {
                    Text("Mic").tag("mic")
                    Text("System").tag("system")
                    Text("Both").tag("both")
                }
                .pickerStyle(.segmented)
                Picker("Model", selection: $m.model) {
                    ForEach(m.languageModels, id: \.self) { Text($0).tag($0) }
                }
                if m.language != "en" {
                    Text("Models that support \(m.languageName). Choose the language when you start a session.")
                        .font(.caption).foregroundStyle(.secondary)
                }
                Picker("Microphone", selection: $m.device) {
                    Text("System default").tag("")
                    ForEach(m.devices, id: \.self) { Text($0).tag($0) }
                }
                .disabled(m.source == "system")
                if m.running {
                    Text("Stop recording to change these.").font(.caption).foregroundStyle(.secondary)
                } else if m.source == "both" {
                    Text("Use headphones so the mic doesn't pick up the other side.")
                        .font(.caption).foregroundStyle(.secondary)
                }
            }
            .disabled(m.running)

            Section("Your profile") {
                HStack {
                    Image(systemName: m.profileLoaded ? "person.crop.circle.badge.checkmark" : "person.crop.circle.badge.questionmark")
                        .foregroundStyle(m.profileLoaded ? Color.green : Color.secondary)
                        .font(.title3)
                    VStack(alignment: .leading, spacing: 2) {
                        Text(m.profileLoaded ? "Loaded · \(m.profileWords) words" : "No profile yet")
                        Text(m.profileLoaded ? "Behavioural answers use your background."
                                             : "Answers stay general until you add your background.")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                }
                HStack {
                    Button("Import CV…", action: m.importCV)
                    Button("Edit…", action: m.editProfile)
                    Button("Reload", action: m.reloadProfile)
                    if m.profileBusy { ProgressView().controlSize(.small) }
                }
                .disabled(m.profileBusy)
                VStack(alignment: .leading, spacing: 6) {
                    Text("Finish it with ChatGPT or Claude").font(.caption.weight(.semibold))
                    Text("Copy a prompt with your CV and profile. The model interviews you to fill the gaps, then gives back the finished profile.")
                        .font(.caption).foregroundStyle(.secondary)
                    HStack {
                        Button {
                            m.copyProfilePrompt()
                        } label: { Label("Copy prompt", systemImage: "doc.on.clipboard") }
                        Button {
                            m.pasteProfileResult()
                        } label: { Label("Paste result", systemImage: "arrow.down.doc") }
                    }
                }
                .disabled(m.profileBusy)
                if !m.profileMessage.isEmpty {
                    Text(m.profileMessage).font(.caption).foregroundStyle(.secondary)
                }
                if !m.profileToAdd.isEmpty {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("To complete it, add to your profile:").font(.caption.weight(.semibold))
                        ForEach(m.profileToAdd, id: \.self) { q in
                            Label(q, systemImage: "plus.circle").font(.caption)
                        }
                    }
                    .foregroundStyle(.secondary)
                }
                Text("Importing sends your CV to your AI provider (OpenCode) to organise it; the profile is then sent with each question. The copied prompt contains your CV, so only paste it into a service you trust with it.")
                    .font(.caption2).foregroundStyle(.tertiary)
            }
        }
        .formStyle(.grouped)
        .frame(width: 380)
        .padding(.vertical, 4)
    }
}

// MARK: - sidebar

struct Sidebar: View {
    @EnvironmentObject var m: AppModel
    @State private var question = ""

    var body: some View {
        VStack(spacing: 0) {
            if m.cards.isEmpty {
                VStack(spacing: 6) {
                    Spacer()
                    Image(systemName: "questionmark.bubble").font(.title).foregroundStyle(.tertiary)
                    Text("No questions yet").foregroundStyle(.secondary)
                    Spacer()
                }
                .frame(maxWidth: .infinity)
            } else {
                List(selection: $m.selectedCard) {
                    Section("Questions") {
                        ForEach(m.cards) { card in
                            QuestionRow(card: card).tag(card.id)
                        }
                    }
                }
                .listStyle(.sidebar)
            }
            Divider()
            HStack(spacing: 6) {
                TextField("Ask a question…", text: $question)
                    .textFieldStyle(.roundedBorder)
                    .onSubmit(send)
                Button(action: send) { Image(systemName: "arrow.up.circle.fill").font(.title3) }
                    .buttonStyle(.borderless)
                    .disabled(question.trimmingCharacters(in: .whitespaces).isEmpty || !m.assistantOn)
                    .help(m.assistantOn ? "Ask" : "Turn on Answers first")
            }
            .padding(10)
        }
    }

    func send() {
        guard m.assistantOn else { return }
        m.ask(question)
        question = ""
    }
}

struct QuestionRow: View {
    let card: Card

    var body: some View {
        HStack(alignment: .top, spacing: 8) {
            Group {
                switch card.status {
                case "thinking": ProgressView().controlSize(.mini)
                case "ready": Image(systemName: "checkmark.circle.fill").foregroundStyle(.green)
                default: Image(systemName: "exclamationmark.triangle.fill").foregroundStyle(.orange)
                }
            }
            .frame(width: 16, height: 16)
            VStack(alignment: .leading, spacing: 2) {
                Text(card.question).lineLimit(2).font(.callout)
                Text(speaker(card.source)).font(.caption2).foregroundStyle(.secondary)
            }
        }
        .padding(.vertical, 3)
    }
}

// MARK: - answer

struct AnswerDetail: View {
    @EnvironmentObject var m: AppModel

    var body: some View {
        if let id = m.selectedCard, let card = m.cards.first(where: { $0.id == id }) {
            ScrollView {
                AnswerView(card: card)
                    .frame(maxWidth: 820, alignment: .leading)
                    .padding(.horizontal, 28).padding(.vertical, 18)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
        } else {
            EmptyState()
        }
    }
}

struct EmptyState: View {
    @EnvironmentObject var m: AppModel

    var body: some View {
        VStack(spacing: 14) {
            Image(systemName: m.assistantOn ? "waveform.and.mic" : "sparkles")
                .font(.system(size: 44, weight: .light)).foregroundStyle(.tertiary)
            Text(title).font(.title2.weight(.semibold))
            Text(detail).foregroundStyle(.secondary).multilineTextAlignment(.center).frame(maxWidth: 420)
            if !m.assistantOn {
                Button("Turn on Answers") { m.setAssistant(true) }.buttonStyle(.borderedProminent)
            } else if !m.running {
                Button("Start listening", action: m.toggleRecording).buttonStyle(.borderedProminent)
            }
        }
        .padding(40)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    var title: String {
        if !m.assistantOn { return "Answers are off" }
        return m.running ? "Listening for a question" : "Ready"
    }

    var detail: String {
        if !m.assistantOn {
            return "Turn on Answers to get a short answer, key points and an example for each technical question."
        }
        let who = m.listenTo.map(speaker).joined(separator: " or ")
        return "Questions asked by \(who.isEmpty ? "the speaker" : who) are answered here automatically. You can also type one in the sidebar."
    }
}

struct AnswerView: View {
    @EnvironmentObject var m: AppModel
    let card: Card

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            VStack(alignment: .leading, spacing: 4) {
                Text(speaker(card.source).uppercased()).font(.caption.weight(.semibold))
                    .foregroundStyle(.secondary).tracking(0.6)
                Text(card.question).font(.system(size: 22, weight: .bold)).textSelection(.enabled)
            }

            switch card.status {
            case "thinking":
                HStack(spacing: 10) {
                    ProgressView().controlSize(.small)
                    Text("Thinking…").font(.title3).foregroundStyle(.secondary)
                }
            case "error":
                Label(card.error, systemImage: "exclamationmark.triangle.fill")
                    .foregroundStyle(.orange).textSelection(.enabled)
            default:
                Text(card.shortAnswer)
                    .font(.system(size: 17)).lineSpacing(4).textSelection(.enabled)
                    .fixedSize(horizontal: false, vertical: true)

                if !card.gap.isEmpty { GapHint(card: card) }

                if !card.keyPoints.isEmpty {
                    SectionBox(title: "Key points", icon: "list.bullet") {
                        VStack(alignment: .leading, spacing: 7) {
                            ForEach(Array(card.keyPoints.enumerated()), id: \.offset) { i, p in
                                HStack(alignment: .firstTextBaseline, spacing: 12) {
                                    Text("\(i + 1)").font(.caption.weight(.bold).monospacedDigit())
                                        .foregroundStyle(.white)
                                        .frame(width: 19, height: 19)
                                        .background(Circle().fill(Color.accentColor))
                                        .alignmentGuide(.firstTextBaseline) { $0[VerticalAlignment.center] + 5 }
                                    Text(p).font(.system(size: 15)).textSelection(.enabled)
                                }
                            }
                        }
                    }
                }

                if !card.example.isEmpty { CodeBlock(code: card.example) }

                YourAnswer(card: card)

                if !card.model.isEmpty {
                    Text("\(card.model) · \(String(format: "%.1f", card.seconds)) s")
                        .font(.caption2).foregroundStyle(.tertiary)
                }
            }
        }
    }
}

/// What the profile is missing for this answer, with a way to add it.
struct GapHint: View {
    @EnvironmentObject var m: AppModel
    let card: Card

    var body: some View {
        HStack(spacing: 10) {
            Image(systemName: "lightbulb").foregroundStyle(.yellow)
            Text(card.gap).font(.callout).foregroundStyle(.secondary)
            Spacer(minLength: 8)
            Button("Add to profile…") { m.noteCard = card }.controlSize(.small)
        }
        .padding(.horizontal, 12).padding(.vertical, 8)
        .background(RoundedRectangle(cornerRadius: 8).fill(Color.yellow.opacity(0.08)))
        .overlay(RoundedRectangle(cornerRadius: 8).stroke(Color.yellow.opacity(0.25)))
    }
}

struct AddNoteSheet: View {
    @EnvironmentObject var m: AppModel
    let card: Card
    @State private var text = ""
    @State private var saving = false

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Add to your profile").font(.title2.weight(.semibold))
            VStack(alignment: .leading, spacing: 3) {
                Text(card.question).font(.headline)
                Text(card.gap).font(.callout).foregroundStyle(.secondary)
            }
            TextEditor(text: $text)
                .font(.body).frame(height: 120)
                .overlay(RoundedRectangle(cornerRadius: 6).stroke(Color.secondary.opacity(0.25)))
            Text("Write only what's true, in a sentence or two: what happened, what you did, the "
                 + "result. It's saved to your profile, and this question is answered again.")
                .font(.caption).foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
            HStack {
                if saving { ProgressView().controlSize(.small) }
                Spacer()
                Button("Cancel") { m.noteCard = nil }.keyboardShortcut(.cancelAction)
                Button("Save & re-answer") {
                    saving = true
                    Task {
                        if await m.addNote(topic: card.question, text: text, cardId: card.id) {
                            m.noteCard = nil
                        }
                        saving = false
                    }
                }
                .keyboardShortcut(.defaultAction).buttonStyle(.borderedProminent)
                .disabled(saving || text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            }
        }
        .padding(20)
        .frame(width: 480)
    }
}

struct SectionBox<Content: View>: View {
    let title: String
    let icon: String
    @ViewBuilder var content: Content

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Label(title, systemImage: icon).font(.caption.weight(.semibold)).foregroundStyle(.secondary)
            content
        }
        .padding(13)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(RoundedRectangle(cornerRadius: 12).fill(Color.primary.opacity(0.045)))
    }
}

struct CodeBlock: View {
    let code: String
    @State private var copied = false

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack {
                Label("Python", systemImage: "chevron.left.forwardslash.chevron.right")
                    .font(.caption.weight(.semibold)).foregroundStyle(.secondary)
                Spacer()
                Button {
                    NSPasteboard.general.clearContents()
                    NSPasteboard.general.setString(code, forType: .string)
                    copied = true
                    DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) { copied = false }
                } label: {
                    Label(copied ? "Copied" : "Copy", systemImage: copied ? "checkmark" : "doc.on.doc")
                        .font(.caption)
                }
                .buttonStyle(.borderless)
            }
            .padding(.horizontal, 12).padding(.vertical, 6)
            Divider()
            ScrollView(.horizontal, showsIndicators: false) {
                Text(code).font(.system(size: 13, design: .monospaced)).lineSpacing(2)
                    .textSelection(.enabled).padding(12)
            }
        }
        .background(RoundedRectangle(cornerRadius: 12).fill(Color(nsColor: .textBackgroundColor)))
        .overlay(RoundedRectangle(cornerRadius: 12).stroke(Color.primary.opacity(0.1)))
    }
}

struct YourAnswer: View {
    @EnvironmentObject var m: AppModel
    let card: Card

    var body: some View {
        if !card.spoken.isEmpty {
            SectionBox(title: "Your answer", icon: "person.wave.2") {
                VStack(alignment: .leading, spacing: 12) {
                    Text(card.spoken.joined(separator: " ")).font(.system(size: 15))
                        .foregroundStyle(.secondary).textSelection(.enabled)
                    if !card.covered.isEmpty || !card.missed.isEmpty {
                        FlowChips(items: card.covered, icon: "checkmark", color: .green)
                        FlowChips(items: card.missed, icon: "xmark", color: .orange)
                        if !card.tip.isEmpty {
                            Label(card.tip, systemImage: "lightbulb").font(.callout)
                        }
                    } else {
                        Button {
                            m.grade(card.id)
                        } label: {
                            Label(card.feedbackStatus == "thinking" ? "Checking…" : "Check my answer",
                                  systemImage: "checklist")
                        }
                        .disabled(card.feedbackStatus == "thinking")
                    }
                }
            }
        }
    }
}

struct FlowChips: View {
    let items: [String]
    let icon: String
    let color: Color

    var body: some View {
        if !items.isEmpty {
            VStack(alignment: .leading, spacing: 6) {
                ForEach(items, id: \.self) { item in
                    Label(item, systemImage: icon)
                        .font(.callout)
                        .padding(.horizontal, 10).padding(.vertical, 4)
                        .background(Capsule().fill(color.opacity(0.14)))
                        .foregroundStyle(color)
                }
            }
        }
    }
}

// MARK: - transcript

struct CaptionBar: View {
    @EnvironmentObject var m: AppModel

    var body: some View {
        let recent = Array(m.lines.suffix(m.partials.isEmpty ? 2 : 1))
        let partial = m.partials.values.sorted { $0.source < $1.source }.first
        HStack(alignment: .center, spacing: 14) {
            VStack(alignment: .leading, spacing: 3) {
                if recent.isEmpty && partial == nil {
                    Text(m.running ? "Listening…" : "Live captions appear here while recording.")
                        .foregroundStyle(.tertiary)
                }
                ForEach(recent) { CaptionLine(line: $0, partial: false) }
                if let partial { CaptionLine(line: partial, partial: true) }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            VStack(alignment: .trailing, spacing: 3) {
                if m.running && m.finalLatency != "–" {
                    Text(m.finalLatency).font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                        .help("Time from end of speech to final text (p50)")
                }
                if m.assistantOn {
                    Label(m.assistantHealthy ? "OpenCode" : "OpenCode offline",
                          systemImage: m.assistantHealthy ? "checkmark.circle" : "exclamationmark.triangle")
                        .font(.caption)
                        .foregroundStyle(m.assistantHealthy ? Color.secondary : Color.orange)
                        .help(m.assistantHealth)
                }
                ForEach(m.sourceStates.keys.sorted().filter { m.sourceStates[$0] != "running" }, id: \.self) {
                    Text("\(speaker($0)): \(m.sourceStates[$0] ?? "")").font(.caption).foregroundStyle(.orange)
                }
                if !m.error.isEmpty {
                    Text(m.error).font(.caption).foregroundStyle(.red).lineLimit(1).help(m.error)
                }
            }
        }
        .font(.callout)
        .padding(.horizontal, 20).padding(.vertical, 10)
        .frame(minHeight: 58)
        .background(.bar)
        .overlay(alignment: .top) { Divider() }
    }
}

struct CaptionLine: View {
    let line: Line
    let partial: Bool

    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: 8) {
            Text(speaker(line.source)).font(.caption.weight(.bold))
                .foregroundStyle(line.source == "mic" ? Color.teal : Color.purple)
                .frame(width: 36, alignment: .leading)
            if partial {
                (Text(line.stable.isEmpty ? "" : line.stable + " ") + Text(line.unstable).foregroundColor(.secondary))
                    .lineLimit(1).truncationMode(.head)
            } else {
                Text(line.text).lineLimit(1).truncationMode(.head)
            }
        }
    }
}

struct TranscriptInspector: View {
    @EnvironmentObject var m: AppModel

    var body: some View {
        ScrollViewReader { proxy in
            List {
                Section("Transcript") {
                    ForEach(m.lines) { line in
                        VStack(alignment: .leading, spacing: 2) {
                            HStack(spacing: 6) {
                                Text(speaker(line.source)).font(.caption.weight(.bold))
                                    .foregroundStyle(line.source == "mic" ? Color.teal : Color.purple)
                                Text(String(format: "%02d:%02d", Int(line.start) / 60, Int(line.start) % 60))
                                    .font(.caption2.monospacedDigit()).foregroundStyle(.tertiary)
                            }
                            Text(line.text).textSelection(.enabled)
                        }
                        .padding(.vertical, 2)
                    }
                    ForEach(m.partials.values.sorted { $0.source < $1.source }) { p in
                        Text(p.stable + " " + p.unstable).foregroundStyle(.secondary)
                    }
                    Color.clear.frame(height: 1).id("bottom").listRowSeparator(.hidden)
                }
            }
            .onChange(of: m.lines.count) { proxy.scrollTo("bottom") }
        }
    }
}


// MARK: - new session

struct NewSessionSheet: View {
    @EnvironmentObject var m: AppModel

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            Text("New session").font(.title2.weight(.semibold)).padding([.horizontal, .top], 20)
            Text("Answers are tailored to this interview. Leave fields empty for general practice.")
                .font(.callout).foregroundStyle(.secondary).padding(.horizontal, 20).padding(.top, 2)
            Form {
                TextField("Title", text: $m.sessionTitle, prompt: Text("e.g. Mock interview, round 2"))
                TextField("Company", text: $m.company, prompt: Text("optional"))
                TextField("Role", text: $m.role, prompt: Text("optional"))
                Picker("Interview language", selection: $m.language) {
                    ForEach(m.languages) { Text($0.name).tag($0.code) }
                }
                .onChange(of: m.language) { m.fitModelToLanguage(changed: true) }
                if m.language != "en" {
                    Text("Transcribed with \(m.model); answers are written in \(m.languageName) to say out loud.")
                        .font(.caption).foregroundStyle(.secondary)
                }
                VStack(alignment: .leading, spacing: 4) {
                    Text("Job description")
                    TextEditor(text: $m.jobDescription)
                        .font(.callout).frame(height: 110)
                        .overlay(RoundedRectangle(cornerRadius: 6).stroke(Color.secondary.opacity(0.25)))
                    Text("Optional. Paste the posting so answers stress what this role asks for.")
                        .font(.caption).foregroundStyle(.secondary)
                }
                PrepSection()
                Toggle("Save this session", isOn: $m.saveSession)
                Toggle("Record audio", isOn: $m.saveAudio).disabled(!m.saveSession)
                if m.saveSession && m.saveAudio && m.source != "mic" {
                    Text("Recording the other side of a call may require their consent.")
                        .font(.caption).foregroundStyle(.orange)
                }
            }
            .formStyle(.grouped)
            .frame(height: m.prepGaps.isEmpty ? 470 : 600)
            HStack {
                Button("Clear fields") {
                    m.sessionTitle = ""; m.company = ""; m.role = ""; m.jobDescription = ""
                }
                Spacer()
                Button("Cancel") { m.showNewSession = false }.keyboardShortcut(.cancelAction)
                Button("Start") { m.startSession() }
                    .keyboardShortcut(.defaultAction).buttonStyle(.borderedProminent)
            }
            .padding(20)
        }
        .frame(width: 560)
    }
}

/// Pre-interview check: likely questions the profile cannot answer yet, filled in here.
struct PrepSection: View {
    @EnvironmentObject var m: AppModel

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Button {
                    m.checkProfile()
                } label: {
                    Label("Check my profile for this interview", systemImage: "checklist")
                }
                .disabled(m.prepBusy)
                if m.prepBusy { ProgressView().controlSize(.small) }
            }
            if !m.prepMessage.isEmpty {
                Text(m.prepMessage).font(.caption).foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            ForEach($m.prepGaps) { $gap in
                VStack(alignment: .leading, spacing: 3) {
                    Text(gap.question).font(.callout.weight(.semibold))
                    Text(gap.missing + (gap.hint.isEmpty ? "" : " · " + gap.hint))
                        .font(.caption).foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                    TextField("Your answer (only what's true)", text: $gap.answer, axis: .vertical)
                        .lineLimit(2...4).textFieldStyle(.roundedBorder)
                }
                .padding(.vertical, 2)
            }
            if !m.prepGaps.isEmpty {
                Button("Save answers to profile") { m.savePrepAnswers() }
                    .disabled(m.prepBusy || m.prepGaps.allSatisfy {
                        $0.answer.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty })
            }
        }
    }
}
