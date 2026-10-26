// Reference masks for rlx-subject-lift parity: the vendor's subject lifting via the public Vision API.
// Build: swiftc -O scripts/vision_subject_lift.swift -o /tmp/vision_subject_lift
// Run:   /tmp/vision_subject_lift OUT_DIR image.png …  → OUT_DIR/<name>-vision.png (masked RGBA),
//        OUT_DIR/<name>-vision-mask.png (soft mask, image resolution). macOS 14+.
import Foundation
import Vision
import CoreImage
import ImageIO
import UniformTypeIdentifiers

func writePNG(_ ci: CIImage, _ url: URL) throws {
    let ctx = CIContext()
    guard let cg = ctx.createCGImage(ci, from: ci.extent, format: .RGBA8, colorSpace: CGColorSpace(name: CGColorSpace.sRGB)!) else { throw NSError(domain: "cg", code: 1) }
    let dst = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil)!
    CGImageDestinationAddImage(dst, cg, nil)
    CGImageDestinationFinalize(dst)
}

let args = CommandLine.arguments
let outDir = URL(fileURLWithPath: args[1])
for path in args.dropFirst(2) {
    let url = URL(fileURLWithPath: path)
    let name = url.deletingPathExtension().lastPathComponent
    let handler = VNImageRequestHandler(url: url)
    let req = VNGenerateForegroundInstanceMaskRequest()
    do {
        try handler.perform([req])
        guard let obs = req.results?.first else { print("\(name): no subject found"); continue }
        let inst = obs.allInstances
        print("\(name): \(inst.count) instance(s)")
        let masked = try obs.generateMaskedImage(ofInstances: inst, from: handler, croppedToInstancesExtent: false)
        try writePNG(CIImage(cvPixelBuffer: masked), outDir.appendingPathComponent("\(name)-vision.png"))
        let mask = try obs.generateScaledMaskForImage(forInstances: inst, from: handler)
        try writePNG(CIImage(cvPixelBuffer: mask), outDir.appendingPathComponent("\(name)-vision-mask.png"))
    } catch {
        print("\(name): \(error)")
    }
}
