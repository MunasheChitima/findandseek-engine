// findandseek OCR + image-tagging helper — native macOS Vision.
//   VNRecognizeTextRequest  → text (printed/scanned)
//   VNClassifyImageRequest  → scene/object tags (only for low-text, pictorial
//                             images, so text pages don't get noisy labels)
// On-device, Neural-Engine accelerated, zero-egress.
//   ocrtool <img1> [img2 ...]   batch mode
//   ocrtool --server            stream mode (one image path per stdin line)
import Foundation
import Vision
import AppKit

let END  = "\u{01}FINDANDSEEK_OCR_END\u{01}"
let TAGS = "\u{02}TAGS\u{02}"
let TEXT_THRESHOLD = 16   // below this, treat as a picture and classify

func recognize(_ cg: CGImage) -> String {
    let req = VNRecognizeTextRequest()
    req.recognitionLevel = .accurate
    req.usesLanguageCorrection = true
    let handler = VNImageRequestHandler(cgImage: cg, options: [:])
    (try? handler.perform([req]))
    return (req.results ?? [])
        .compactMap { $0.topCandidates(1).first?.string }
        .joined(separator: "\n")
}

func classify(_ cg: CGImage) -> [String] {
    let req = VNClassifyImageRequest()
    let handler = VNImageRequestHandler(cgImage: cg, options: [:])
    (try? handler.perform([req]))
    let obs = (req.results ?? []).filter { $0.confidence > 0.15 }
    return obs.prefix(6).map { $0.identifier }
}

func process(_ path: String) -> (String, [String]) {
    guard let img = NSImage(contentsOfFile: path),
          let cg = img.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
        return ("", [])
    }
    let text = recognize(cg)
    let tags = text.count < TEXT_THRESHOLD ? classify(cg) : []
    return (text, tags)
}

func emit(_ path: String) {
    let (text, tags) = process(path)
    print(text)
    if !tags.isEmpty { print(TAGS + tags.joined(separator: ",")) }
    print(END)
    fflush(stdout)
}

let args = Array(CommandLine.arguments.dropFirst())
if args.first == "--server" {
    while let line = readLine(strippingNewline: true) {
        if line.isEmpty { print(END); fflush(stdout); continue }
        emit(line)
    }
} else {
    for path in args { emit(path) }
}
