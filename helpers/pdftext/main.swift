// pdftext: print a PDF's text using Apple's PDFKit, which keeps word spacing for PDFs where
// pypdf glues words together (custom font encodings). Used for CV import.
//   pdftext FILE.pdf
import Foundation
import PDFKit

guard CommandLine.arguments.count == 2,
      let doc = PDFDocument(url: URL(fileURLWithPath: CommandLine.arguments[1])) else {
    FileHandle.standardError.write("usage: pdftext FILE.pdf (could not open the PDF)\n".data(using: .utf8)!)
    exit(1)
}
print(doc.string ?? "")
