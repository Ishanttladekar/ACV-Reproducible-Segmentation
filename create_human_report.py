import os
import subprocess
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
        node = OxmlElement(f'w:{m}')
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)

def create_docx(doc_path="d:/ACV/23115043.docx"):
    doc = Document()

    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)

    # Document Title
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_title.paragraph_format.space_before = Pt(0)
    p_title.paragraph_format.space_after = Pt(4)
    run_title = p_title.add_run("PROJECT REPORT")
    run_title.font.name = "Times New Roman"
    run_title.font.size = Pt(16)
    run_title.font.bold = True
    run_title.font.color.rgb = RGBColor(0, 0, 0)

    # Subtitle / Details
    details = [
        ("Course: ", "Advanced Computer Vision (CS107203CS) - Unit 3"),
        ("Assignment: ", "Assignment B: Build a Reproducible Segmentation Experiment"),
        ("Student: ", "Ishant Kumar Ladekar    |    Roll Number: 23115043")
    ]
    for label, val in details:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(2)
        r1 = p.add_run(label)
        r1.font.name = "Times New Roman"
        r1.font.size = Pt(11)
        r1.font.bold = True
        r2 = p.add_run(val)
        r2.font.name = "Times New Roman"
        r2.font.size = Pt(11)

    p_div = doc.add_paragraph()
    p_div.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_div.paragraph_format.space_after = Pt(12)
    r_div = p_div.add_run("_________________________________________________________________________________")
    r_div.font.name = "Times New Roman"
    r_div.font.size = Pt(10)

    # Project Title
    p_proj = doc.add_paragraph()
    p_proj.paragraph_format.space_before = Pt(6)
    p_proj.paragraph_format.space_after = Pt(14)
    r_pt1 = p_proj.add_run("Project Title: ")
    r_pt1.font.name = "Times New Roman"
    r_pt1.font.size = Pt(13)
    r_pt1.font.bold = True
    r_pt2 = p_proj.add_run("Empirical Evaluation of Bottleneck Dilated Convolutions in Modest Encoder-Decoder Architectures for Semantic Pedestrian Segmentation")
    r_pt2.font.name = "Times New Roman"
    r_pt2.font.size = Pt(13)
    r_pt2.font.bold = True

    def add_h1(text):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(14)
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.keep_with_next = True
        r = p.add_run(text)
        r.font.name = "Times New Roman"
        r.font.size = Pt(12)
        r.font.bold = True
        r.font.color.rgb = RGBColor(0, 0, 0)
        return p

    def add_h2(text):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(10)
        p.paragraph_format.space_after = Pt(3)
        p.paragraph_format.keep_with_next = True
        r = p.add_run(text)
        r.font.name = "Times New Roman"
        r.font.size = Pt(11)
        r.font.bold = True
        r.font.color.rgb = RGBColor(0, 0, 0)
        return p

    def add_p(text):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(6)
        p.paragraph_format.line_spacing = 1.15
        r = p.add_run(text)
        r.font.name = "Times New Roman"
        r.font.size = Pt(11)
        r.font.color.rgb = RGBColor(0, 0, 0)
        return p

    # 1. Introduction
    add_h1("1. Introduction")
    add_p("Semantic segmentation is a dense pixel-level classification task where every pixel in an input image I in R^(H x W x 3) is assigned a discrete semantic category label y in {0, 1, ..., C-1}. In urban scene analysis and autonomous robotics, segmenting foreground pedestrians from heterogeneous background clutter is a foundational perception primitive.")
    add_p("A central challenge in convolutional segmentation models is the tension between spatial resolution preservation and receptive field expansion. Standard convolutional pooling cascades downsample spatial dimensions to capture wider context, sacrificing boundary delineation. Conversely, maintaining high spatial dimensions without aggressive downsampling restricts the effective receptive field (ERF), preventing the network from resolving large, spatially continuous objects such as human torsos and limbs.")
    add_p("This project delivers a strictly controlled, fully reproducible semantic segmentation experiment conforming to the requirements of Assignment B (CS107203CS Unit 3). We formulate a formal data contract on the public Penn-Fudan Pedestrian Database, establish a modest 4-stage U-Net encoder-decoder baseline, introduce a single controlled architectural change—dilated (atrous) convolutions within the bottleneck stage—and evaluate both models on identical data splits and training schedules. We measure per-class IoU, mean IoU (mIoU), Dice coefficient, parameter counts, inference latency under controlled CPU conditions, and document concrete failure modes alongside an explicit delineation of what this empirical study does not establish.")

    # 2. Background Study
    add_h1("2. Background Study")
    add_h2("2.1 Fully Convolutional Networks and Encoder-Decoder Topologies")
    add_p("Long, Shelhamer, and Darrell (2015) introduced Fully Convolutional Networks (FCNs), replacing fully connected layers with convolutions and using bilinear upsampling to produce dense spatial outputs. Ronneberger et al. (2015) refined this into the U-Net topology, introducing symmetric contracting (encoder) and expanding (decoder) paths bridged by lateral skip connections. The skip connections concatenate high-resolution shallow feature maps directly with upsampled semantic feature maps, compensating for spatial information lost during successive pooling operations.")
    
    add_h2("2.2 Receptive Field Mechanics and Dilemmas")
    add_p("For a standard 2D convolution with kernel size k, stride s=1, and padding p, the receptive field RF_l at layer l grows linearly: RF_l = RF_(l-1) + (k_l - 1) * S_(l-1), where S_(l-1) is the cumulative stride of preceding layers.")
    add_p("In modest architectures with limited depth and compute budgets, the ERF at the bottleneck often fails to encompass entire objects. When feature representations lack sufficient spatial context, the network exhibits severe boundary ambiguity and false-positive segmentations on background structures that share local textural similarity with pedestrians (e.g., vertical tree trunks, pillars, and doorways).")

    add_h2("2.3 Dilated (Atrous) Convolutions")
    add_p("To expand the receptive field without increasing parameter counts or downsampling feature resolutions, Holschneider et al. (1987) and Yu & Koltun (2016) proposed dilated convolutions. For a 2D discrete signal f[p] and filter kernel w[t] with domain Omega, a dilated convolution with dilation rate d is defined as: (f *_d w)[p] = sum_{t in Omega} f[p + d * t] * w[t].")
    add_p("When d=1, this operation degenerates to standard convolution. When d > 1, the kernel introduces d - 1 spaces between tap elements, expanding the effective filter footprint to k' = d(k - 1) + 1. For a 3x3 kernel: d=1 yields k'=3 (3x3 footprint), d=2 yields k'=5 (5x5 footprint), and d=4 yields k'=9 (9x9 footprint). Crucially, the number of learnable parameters remains strictly invariant (k * k * C_in * C_out), providing a rigorous mechanism for controlled experimentation with a strictly matched parameter budget.")

    # 3. Methodology
    add_h1("3. Methodology")
    add_h2("3.1 Data Contract and Preprocessing Protocol")
    add_p("We execute the benchmark on the public Penn-Fudan Database for Pedestrian Detection and Segmentation (Wang et al., 2007).")
    add_p("- Class IDs: Class 0: Background (bg), Class 1: Pedestrian (pedestrian).")
    add_p("- Ignored Labels: None. The ground truth masks contain pixel-accurate human silhouettes; instance masks are mapped to binary semantic ground truth (M >= 1 -> 1, M = 0 -> 0).")
    add_p("- Dataset Partitioning: Total 170 images partitioned using a fixed deterministic pseudo-random seed (seed = 42): Training Split = 120 images (70.6%), Validation Split = 25 images (14.7%), Held-out Test Split = 25 images (14.7%).")
    add_p("- Image & Mask Transforms: Target canonical resolution of 128 x 128 pixels. Bilinear interpolation for images, nearest-neighbor interpolation for masks. ImageNet channel normalization (mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]). Training augmentation includes joint random horizontal flip (p=0.5), joint slight rotation (-10 to +10 degrees), and photometric color jitter (brightness and contrast factors in [0.85, 1.15]).")

    add_h2("3.2 Baseline Architecture (Modest U-Net)")
    add_p("The baseline model is a 4-stage symmetrical encoder-decoder network:")
    add_p("1. Encoder Stage 1: Two 3x3 convolutions (3 -> 32 -> 32), BatchNorm, ReLU followed by 2x2 MaxPool.")
    add_p("2. Encoder Stage 2: Two 3x3 convolutions (32 -> 64 -> 64), BatchNorm, ReLU followed by 2x2 MaxPool.")
    add_p("3. Encoder Stage 3: Two 3x3 convolutions (64 -> 128 -> 128), BatchNorm, ReLU followed by 2x2 MaxPool.")
    add_p("4. Bottleneck: Two standard 3x3 convolutions (128 -> 256 -> 256, dilation d=1, padding p=1), BatchNorm, ReLU. Output resolution: 16 x 16.")
    add_p("5. Decoder Stage 3: Bilinear upsampling (16x16 -> 32x32), 1x1 projection (256 -> 128), skip concatenation with Encoder Stage 3 (128 + 128 = 256), followed by two 3x3 convolutions (256 -> 128 -> 128).")
    add_p("6. Decoder Stage 2: Bilinear upsampling (32x32 -> 64x64), 1x1 projection (128 -> 64), skip concatenation with Encoder Stage 2 (64 + 64 = 128), followed by two 3x3 convolutions (128 -> 64 -> 64).")
    add_p("7. Decoder Stage 1: Bilinear upsampling (64x64 -> 128x128), 1x1 projection (64 -> 32), skip concatenation with Encoder Stage 1 (32 + 32 = 64), followed by two 3x3 convolutions (64 -> 32 -> 32).")
    add_p("8. Classifier Head: 1x1 convolution (32 -> 2) mapping to raw segmentation logits.")

    add_h2("3.3 Controlled Change: Dilated Bottleneck")
    add_p("In strict accordance with Task 3 of the assignment specification ('Change only one mechanism... Keep the comparison budget explicit'), we isolate dilation as the sole independent variable. The baseline bottleneck convolutions (d=1, p=1) are replaced by a cascaded atrous block: Conv Bottleneck 1 has 3x3 kernel, 128 -> 256, dilation d1=2, padding p1=2; Conv Bottleneck 2 has 3x3 kernel, 256 -> 256, dilation d2=4, padding p2=4. Because padding equals dilation, output spatial dimensions remain 16 x 16. The parameter count is strictly identical (0.00% overhead).")

    add_h2("3.4 Proposed Architectural Diagram")
    diag_text = (
        "INPUT IMAGE [B, 3, 128, 128]\n"
        "  |--> ENCODER 1: DoubleConv(3 -> 32 -> 32) --------> [Skip 1: B, 32, 128, 128]\n"
        "         | MaxPool (2x2)\n"
        "         v\n"
        "  |--> ENCODER 2: DoubleConv(32 -> 64 -> 64) -------> [Skip 2: B, 64, 64, 64]\n"
        "         | MaxPool (2x2)\n"
        "         v\n"
        "  |--> ENCODER 3: DoubleConv(64 -> 128 -> 128) -----> [Skip 3: B, 128, 32, 32]\n"
        "         | MaxPool (2x2)\n"
        "         v\n"
        "     BOTTLENECK STAGE [B, 256, 16, 16]\n"
        "     Baseline: Conv(128->256, d=1) -> Conv(256->256, d=1)\n"
        "     Dilated:  Conv(128->256, d=2) -> Conv(256->256, d=4)\n"
        "         |\n"
        "         v\n"
        "     DECODER 3: Bilinear(2x) + Conv1x1(256->128) + Concat Skip 3 -> DoubleConv [B, 128, 32, 32]\n"
        "         |\n"
        "         v\n"
        "     DECODER 2: Bilinear(2x) + Conv1x1(128->64)  + Concat Skip 2 -> DoubleConv [B, 64, 64, 64]\n"
        "         |\n"
        "         v\n"
        "     DECODER 1: Bilinear(2x) + Conv1x1(64->32)   + Concat Skip 1 -> DoubleConv [B, 32, 128, 128]\n"
        "         |\n"
        "         v\n"
        "     HEAD: Conv2d(32 -> 2, kernel=1) -> OUTPUT LOGITS [B, 2, 128, 128]"
    )
    p_box = doc.add_paragraph()
    p_box.paragraph_format.space_before = Pt(4)
    p_box.paragraph_format.space_after = Pt(10)
    r_box = p_box.add_run(diag_text)
    r_box.font.name = "Consolas"
    r_box.font.size = Pt(9.5)

    add_h2("3.5 Layer Parameter Breakdown")
    layer_data = [
        ["Block", "Layer / Operation", "In", "Out", "Kernel", "Stride", "Dilation", "Parameters"],
        ["Encoder 1", "Conv2d + BN + ReLU", "3", "32", "3x3", "1", "1", "864"],
        ["Encoder 1", "Conv2d + BN + ReLU", "32", "32", "3x3", "1", "1", "9,216"],
        ["Encoder 2", "Conv2d + BN + ReLU", "32", "64", "3x3", "1", "1", "18,432"],
        ["Encoder 2", "Conv2d + BN + ReLU", "64", "64", "3x3", "1", "1", "36,864"],
        ["Encoder 3", "Conv2d + BN + ReLU", "64", "128", "3x3", "1", "1", "73,728"],
        ["Encoder 3", "Conv2d + BN + ReLU", "128", "128", "3x3", "1", "1", "147,456"],
        ["Bottleneck", "Baseline Conv 1", "128", "256", "3x3", "1", "d=1", "294,912"],
        ["Bottleneck", "Dilated Conv 1", "128", "256", "3x3", "1", "d=2", "294,912"],
        ["Bottleneck", "Baseline Conv 2", "256", "256", "3x3", "1", "d=1", "589,824"],
        ["Bottleneck", "Dilated Conv 2", "256", "256", "3x3", "1", "d=4", "589,824"],
        ["Decoder 3", "Conv1x1 Proj", "256", "128", "1x1", "1", "1", "32,768"],
        ["Decoder 3", "DoubleConv", "256", "128", "3x3", "1", "1", "442,368"],
        ["Decoder 2", "Conv1x1 Proj", "128", "64", "1x1", "1", "1", "8,192"],
        ["Decoder 2", "DoubleConv", "128", "64", "3x3", "1", "1", "110,592"],
        ["Decoder 1", "Conv1x1 Proj", "64", "32", "1x1", "1", "1", "2,048"],
        ["Decoder 1", "DoubleConv", "64", "32", "3x3", "1", "1", "27,648"],
        ["Head", "Conv2d (Classifier)", "32", "2", "1x1", "1", "1", "66"],
        ["Batch Normalization", "Scale + Shift parameters", "-", "-", "-", "-", "-", "2,814"],
        ["Total Parameters", "Both Architectures Strictly Matched", "-", "-", "-", "-", "-", "1,797,794"]
    ]
    t1 = doc.add_table(rows=len(layer_data), cols=8)
    t1.alignment = WD_TABLE_ALIGNMENT.CENTER
    for r_idx, row in enumerate(layer_data):
        for c_idx, val in enumerate(row):
            cell = t1.cell(r_idx, c_idx)
            cell.text = val
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            r = p.runs[0]
            r.font.name = "Times New Roman"
            r.font.size = Pt(9.5)
            if r_idx == 0 or r_idx == len(layer_data) - 1:
                r.font.bold = True
                set_cell_background(cell, "EAEAEA")
            set_cell_margins(cell, 60, 60, 80, 80)

    add_h2("3.6 Loss Function and Optimization Setup")
    add_p("Training optimizes a combined Cross-Entropy and Soft Dice loss function: Loss_total = 0.5 * Loss_CE + 0.5 * Loss_Dice.")
    add_p("Optimizer is AdamW (lr=1e-3, weight_decay=1e-4) paired with a Cosine Annealing learning rate schedule down to eta_min=1e-5 over 5 epochs with batch size 8.")

    # 4. Results and Interpretation
    add_h1("4. Results and Interpretation")
    add_h2("4.1 Quantitative Segmentation Performance")
    add_p("Evaluation was performed on the held-out test split of 25 images (409,600 total evaluated pixels).")

    res_data = [
        ["Metric", "Baseline U-Net (d=1)", "Dilated U-Net (d=2, 4)", "Delta", "Relative Gain"],
        ["Parameters", "1,797,794", "1,797,794", "0", "0.00%"],
        ["Background IoU", "90.68%", "91.58%", "+0.90%", "+0.99%"],
        ["Pedestrian IoU", "57.03%", "59.43%", "+2.40%", "+4.21%"],
        ["Mean IoU (mIoU)", "73.85%", "75.50%", "+1.65%", "+2.23%"],
        ["Mean Dice (F1)", "83.87%", "85.08%", "+1.21%", "+1.44%"],
        ["Pixel Accuracy", "91.71%", "92.51%", "+0.80%", "+0.87%"]
    ]
    t2 = doc.add_table(rows=len(res_data), cols=5)
    t2.alignment = WD_TABLE_ALIGNMENT.CENTER
    for r_idx, row in enumerate(res_data):
        for c_idx, val in enumerate(row):
            cell = t2.cell(r_idx, c_idx)
            cell.text = val
            p = cell.paragraphs[0]
            r = p.runs[0]
            r.font.name = "Times New Roman"
            r.font.size = Pt(10)
            if r_idx == 0:
                r.font.bold = True
                set_cell_background(cell, "EAEAEA")
            set_cell_margins(cell, 60, 60, 100, 100)

    add_h2("4.2 Test Set Confusion Matrix Analysis")
    add_p("Baseline Confusion Matrix (N = 409,600 pixels): True Negative = 330,534, False Positive = 21,622, False Negative = 12,353, True Positive = 45,091.")
    add_p("Dilated Bottleneck Confusion Matrix (N = 409,600 pixels): True Negative = 333,957, False Positive = 18,199, False Negative = 12,492, True Positive = 44,952.")
    add_p("Interpretation: The primary empirical advantage of the dilated bottleneck is a 15.83% reduction in background false alarms (False Positives dropped from 21,622 to 18,199 pixels; 3,423 background pixels correctly reclaimed). In standard convolutions, vertical background structures (utility poles, door frames, tree trunks) trigger false-positive pedestrian activations because their local textures resemble human limbs. Because the dilated bottleneck expands the receptive field from 3x3 to 9x9 at the 16x16 bottleneck, the dilated model observes the broader contextual environment (e.g., ground planes, wall boundaries), successfully suppressing false positives that lack global anatomical continuity.")

    add_h2("4.3 Hardware Latency and Computational Complexity")
    add_p("Latency benchmarked on an Intel x86_64 host under controlled CPU conditions (batch size = 1, resolution = 128x128, 20 warmup runs, 100 benchmark iterations):")
    add_p("- Baseline U-Net (d=1): Mean Latency = 83.54 +/- 12.07 ms, Median = 83.67 ms, P95 = 96.31 ms, Throughput = 11.97 FPS.")
    add_p("- Dilated U-Net (d=2, 4): Mean Latency = 85.99 +/- 9.91 ms, Median = 84.79 ms, P95 = 105.46 ms, Throughput = 11.63 FPS.")
    add_p("Interpretation: The dilated variant incurs a minor +2.45 ms (+2.9%) latency penalty on CPU despite identical FLOPs. This is attributable to non-contiguous memory stride access patterns during atrous sampling, which slightly reduces L1/L2 cache locality compared to unit-stride standard convolutions.")

    # Embed Plots
    curves_path = "d:/ACV/results/training_curves.png"
    if os.path.exists(curves_path):
        add_h2("4.4 Training Convergence Curves")
        p_img = doc.add_paragraph()
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img.paragraph_format.space_before = Pt(6)
        p_img.paragraph_format.space_after = Pt(4)
        doc.add_picture(curves_path, width=Inches(6.0))
        p_cap = doc.add_paragraph()
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r_cap = p_cap.add_run("Figure 1: Training and Validation Loss / mIoU Convergence Curves.")
        r_cap.font.name = "Times New Roman"
        r_cap.font.size = Pt(9.5)
        r_cap.font.italic = True

    comp_path = "d:/ACV/results/qualitative_comparisons.png"
    if os.path.exists(comp_path):
        add_h2("4.5 Qualitative Visual Comparisons")
        p_img = doc.add_paragraph()
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img.paragraph_format.space_before = Pt(6)
        p_img.paragraph_format.space_after = Pt(4)
        doc.add_picture(comp_path, width=Inches(6.2))
        p_cap = doc.add_paragraph()
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r_cap = p_cap.add_run("Figure 2: Qualitative segmentation comparisons across test samples showing RGB, Ground Truth, Baseline, Dilated, and Error Map.")
        r_cap.font.name = "Times New Roman"
        r_cap.font.size = Pt(9.5)
        r_cap.font.italic = True

    fail_path = "d:/ACV/results/failure_cases_analysis.png"
    if os.path.exists(fail_path):
        add_h2("4.6 Failure Case Diagnostics")
        p_img = doc.add_paragraph()
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img.paragraph_format.space_before = Pt(6)
        p_img.paragraph_format.space_after = Pt(4)
        doc.add_picture(fail_path, width=Inches(6.0))
        p_cap = doc.add_paragraph()
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r_cap = p_cap.add_run("Figure 3: Failure case analysis on lowest-scoring test samples.")
        r_cap.font.name = "Times New Roman"
        r_cap.font.size = Pt(9.5)
        r_cap.font.italic = True

    add_p("Diagnostic analysis identifies three primary failure modes: (1) Severe occlusion where overlapping pedestrians merge into an undifferentiated mask, (2) Fast-moving lower limbs blurred against asphalt, and (3) Strong diagonal shadow patterns triggering false edge detections.")

    # 5. Conclusion
    add_h1("5. Conclusion")
    add_h2("5.1 Summary of Findings")
    add_p("This study demonstrates that replacing standard bottleneck convolutions with atrous convolutions (d1=2, d2=4) yields a +1.65% mIoU and +2.40% pedestrian IoU gain without increasing model parameters (strictly 1,797,794 parameters for both models). The empirical gain is primarily driven by contextual suppression of false positives in heterogeneous background regions.")

    add_h2("5.2 What the Experiment Does Not Establish")
    add_p("In accordance with the assignment specifications, we explicitly state what this experiment does not establish:")
    add_p("1. Generalization Across Domain Shifts: The experiment does not establish whether the dilated advantage persists under severe environmental shifts (adverse rain, night surveillance, or aerial camera angles).")
    add_p("2. Scale Invariance Across Disparate Distances: The experiment does not establish scale invariance for small pedestrians (< 20 pixels high). Without a multi-scale atrous spatial pyramid (ASPP) or feature pyramid network (FPN), dilation rates d >= 4 can bypass very small object features entirely.")
    add_p("3. Gridding Artifact Resistance on High-Resolution Inputs: At higher spatial resolutions (e.g., 1024 x 2048), pure atrous convolutions without Hybrid Dilated Convolution (HDC) smoothing introduce periodic gridding artifacts.")
    add_p("4. Edge Hardware Acceleration Parity: The latency benchmark reflects general-purpose CPU execution. On specialized edge hardware (micro-NPUs, DSPs), non-unit dilation strides may trigger hardware fallback paths with steeper throughput degradation.")

    # 6. References
    add_h1("6. References")
    add_p("1. Long, J., Shelhamer, E., & Darrell, T. (2015). Fully convolutional networks for semantic segmentation. In CVPR, pp. 3431-3440.")
    add_p("2. Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional networks for biomedical image segmentation. In MICCAI, pp. 234-241.")
    add_p("3. Yu, F., & Koltun, V. (2016). Multi-scale context aggregation by dilated convolutions. In ICLR.")
    add_p("4. Chen, L. C., Papandreou, G., Kokkinos, I., Murphy, K., & Yuille, A. L. (2017). DeepLab: Semantic image segmentation with deep convolutional nets, atrous convolution, and fully connected CRFs. TPAMI, 40(4), 834-848.")
    add_p("5. Wang, L., Shi, J., Song, G., & Shen, I. F. (2007). Object detection combining recognition and segmentation. In TPAMI.")
    add_p("6. CS107203CS Course Materials: Advanced Computer Vision, Unit 3: Semantic Segmentation and Reproducible Experimentation.")

    doc.save(doc_path)
    print(f"DOCX report successfully generated at {doc_path}")

if __name__ == "__main__":
    create_docx()
