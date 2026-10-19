// The app icon, drawn in code: the window is a plain executable (no .app bundle or asset
// catalog), so the Dock icon is set at launch with NSApp.applicationIconImage.
//
// A speech bubble with a waveform (the transcript) and a spark (the AI coach), on an
// indigo-to-blue squircle following the macOS icon grid (824 pt body on a 1024 canvas).

import AppKit

func makeAppIcon(size: CGFloat = 1024) -> NSImage {
    NSImage(size: NSSize(width: size, height: size), flipped: false) { _ in
        let s = size / 1024
        let body = NSRect(x: 100 * s, y: 100 * s, width: 824 * s, height: 824 * s)
        let squircle = NSBezierPath(roundedRect: body, xRadius: 185 * s, yRadius: 185 * s)

        // Soft drop shadow, as on system icons.
        NSGraphicsContext.saveGraphicsState()
        let shadow = NSShadow()
        shadow.shadowColor = NSColor.black.withAlphaComponent(0.3)
        shadow.shadowBlurRadius = 24 * s
        shadow.shadowOffset = NSSize(width: 0, height: -10 * s)
        shadow.set()
        NSColor(calibratedRed: 0.29, green: 0.27, blue: 0.93, alpha: 1).setFill()
        squircle.fill()
        NSGraphicsContext.restoreGraphicsState()

        NSGradient(colors: [
            NSColor(calibratedRed: 0.36, green: 0.30, blue: 0.96, alpha: 1),  // indigo, top
            NSColor(calibratedRed: 0.13, green: 0.55, blue: 1.00, alpha: 1),  // blue, bottom
        ])?.draw(in: squircle, angle: -90)

        // A faint highlight on the upper half for depth.
        NSGraphicsContext.saveGraphicsState()
        squircle.addClip()
        NSGradient(colors: [NSColor.white.withAlphaComponent(0.16), NSColor.white.withAlphaComponent(0)])?
            .draw(in: NSRect(x: body.minX, y: body.midY, width: body.width, height: body.height / 2),
                  angle: -90)
        NSGraphicsContext.restoreGraphicsState()

        // Speech bubble.
        drawSymbol("bubble.fill", in: NSRect(x: 200 * s, y: 215 * s, width: 624 * s, height: 560 * s),
                   color: .white)
        // Waveform inside the bubble.
        drawSymbol("waveform", in: NSRect(x: 318 * s, y: 380 * s, width: 388 * s, height: 270 * s),
                   color: NSColor(calibratedRed: 0.27, green: 0.36, blue: 0.97, alpha: 1))
        // The AI spark, top right.
        drawSymbol("sparkle", in: NSRect(x: 676 * s, y: 664 * s, width: 170 * s, height: 170 * s),
                   color: NSColor(calibratedRed: 1.0, green: 0.84, blue: 0.35, alpha: 1))
        return true
    }
}

/// Draws an SF Symbol in one colour, scaled to fit `rect` and centred in it.
private func drawSymbol(_ name: String, in rect: NSRect, color: NSColor) {
    let config = NSImage.SymbolConfiguration(pointSize: rect.height, weight: .semibold)
        .applying(.init(paletteColors: [color]))
    guard let symbol = NSImage(systemSymbolName: name, accessibilityDescription: nil)?
        .withSymbolConfiguration(config) else { return }
    let scale = min(rect.width / symbol.size.width, rect.height / symbol.size.height)
    let w = symbol.size.width * scale, h = symbol.size.height * scale
    symbol.draw(in: NSRect(x: rect.midX - w / 2, y: rect.midY - h / 2, width: w, height: h))
}

/// Development aid: write the icon to a PNG (OPENINTERVIEW_UI_ICON=path).
func writeAppIcon(to path: String, size: CGFloat = 1024) {
    let image = makeAppIcon(size: size)
    guard let tiff = image.tiffRepresentation, let rep = NSBitmapImageRep(data: tiff),
          let png = rep.representation(using: .png, properties: [:]) else { return }
    try? png.write(to: URL(fileURLWithPath: path))
}
