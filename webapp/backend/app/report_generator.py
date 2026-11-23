"""
PDF Report Generator for Skin Lesion Classification.
Produces clean, professional, publication-quality medical educational reports using ReportLab.
Conforms strictly to non-diagnostic educational standards with required medical disclaimers.
"""

import io
import datetime
import base64
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, KeepTogether, HRFlowable
)
from reportlab.lib.units import inch
from PIL import Image

from .class_info import MEDICAL_DISCLAIMER


def base64_to_pil(base64_str: str) -> Image.Image:
    """Converts a data-URI or raw base64 string to a PIL Image."""
    if "," in base64_str:
        base64_str = base64_str.split(",", 1)[1]
    img_data = base64.b64decode(base64_str)
    return Image.open(io.BytesIO(img_data)).convert("RGB")


def pil_to_reportlab_image(pil_img: Image.Image, max_width: float, max_height: float) -> RLImage:
    """Resizes a PIL image maintaining aspect ratio and wraps into a ReportLab Image."""
    orig_w, orig_h = pil_img.size
    ratio = min(max_width / orig_w, max_height / orig_h)
    new_w = orig_w * ratio
    new_h = orig_h * ratio
    
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=85)
    buf.seek(0)
    return RLImage(buf, width=new_w, height=new_h)


def generate_pdf_report(inference_result: dict) -> bytes:
    """
    Generates a PDF diagnostic summary from an inference result dictionary.
    Returns the generated PDF as raw bytes.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36
    )
    
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#0F172A')
    )
    
    subtitle_style = ParagraphStyle(
        'DocSub',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=12,
        textColor=colors.HexColor('#64748B')
    )
    
    section_heading = ParagraphStyle(
        'SecHead',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=15,
        textColor=colors.HexColor('#1E3A8A'),
        spaceBefore=8,
        spaceAfter=4
    )
    
    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=12,
        textColor=colors.HexColor('#334155')
    )
    
    disclaimer_style = ParagraphStyle(
        'Disclaimer',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=8,
        leading=11,
        textColor=colors.HexColor('#78350F')
    )
    
    story = []
    
    # 1. Header Banner
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header_table = Table([
        [
            Paragraph("<b>Skin Lesion AI</b> &mdash; Classification Analysis", title_style),
            Paragraph(f"<b>Generated:</b> {now_str}<br/><b>Model:</b> ConvNeXt-Tiny (HAM10000)", subtitle_style)
        ]
    ], colWidths=[360, 180])
    header_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('ALIGN', (1, 0), (1, 0), 'RIGHT')
    ]))
    story.append(header_table)
    story.append(Spacer(1, 6))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#3B82F6'), spaceAfter=10))
    
    pred = inference_result.get("predicted_class", {})
    top_3 = inference_result.get("top_3_predictions", [])
    dist = inference_result.get("confidence_distribution", [])
    gradcam_data = inference_result.get("gradcam", {})
    orig_b64 = inference_result.get("original_image_base64")
    
    # 2. Visual Comparison: Original Lesion vs Grad-CAM
    img_cells = []
    if orig_b64:
        try:
            orig_pil = base64_to_pil(orig_b64)
            rl_orig = pil_to_reportlab_image(orig_pil, max_width=2.5*inch, max_height=2.2*inch)
            img_cells.append([Paragraph("<b>Original Uploaded Image</b>", body_style), rl_orig])
        except Exception:
            pass
            
    if gradcam_data.get("available") and gradcam_data.get("image_base64"):
        try:
            cam_pil = base64_to_pil(gradcam_data["image_base64"])
            rl_cam = pil_to_reportlab_image(cam_pil, max_width=2.5*inch, max_height=2.2*inch)
            img_cells.append([Paragraph("<b>Grad-CAM Saliency Overlay</b>", body_style), rl_cam])
        except Exception:
            pass
            
    if img_cells:
        if len(img_cells) == 2:
            img_table = Table([
                [img_cells[0][0], img_cells[1][0]],
                [img_cells[0][1], img_cells[1][1]]
            ], colWidths=[270, 270])
        else:
            img_table = Table([[img_cells[0][0]], [img_cells[0][1]]], colWidths=[540])
        img_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4)
        ]))
        story.append(img_table)
        story.append(Spacer(1, 10))
        
    # 3. Primary Prediction Summary Card
    pred_card = [
        [
            Paragraph("<b>Predicted Category:</b>", body_style),
            Paragraph(f"<b><font size=12 color='#0F172A'>{pred.get('name', 'N/A')}</font></b> ({pred.get('code', '').upper()})", body_style)
        ],
        [
            Paragraph("<b>Classification Confidence:</b>", body_style),
            Paragraph(f"<b><font color='#2563EB' size=11>{pred.get('confidence_percent', 0)}%</font></b> (Softmax score, not clinical probability)", body_style)
        ],
        [
            Paragraph("<b>Clinical Risk Stratification:</b>", body_style),
            Paragraph(f"<b>{pred.get('risk_level', 'N/A')}</b> &bull; {pred.get('type', 'N/A')}", body_style)
        ]
    ]
    pred_table = Table(pred_card, colWidths=[160, 380])
    pred_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F8FAFC')),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#CBD5E1')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#E2E8F0')),
        ('PADDING', (0, 0), (-1, -1), 5),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')
    ]))
    story.append(pred_table)
    story.append(Spacer(1, 8))
    
    # 4. Clinical Context & Description
    if pred.get("description"):
        story.append(Paragraph("<b>Educational Summary & Guidance</b>", section_heading))
        story.append(Paragraph(pred["description"], body_style))
        if pred.get("clinical_guidance"):
            story.append(Spacer(1, 3))
            story.append(Paragraph(f"<b>Recommended Clinical Next Steps:</b> {pred['clinical_guidance']}", body_style))
        story.append(Spacer(1, 8))
        
    # 5. Top 3 Predictions Table
    story.append(Paragraph("<b>Top 3 Differential Candidates</b>", section_heading))
    top3_rows = [["Rank", "Category Name", "Risk Tier", "Confidence Score"]]
    for rank, item in enumerate(top_3, 1):
        top3_rows.append([
            f"#{rank}",
            f"{item['short_name']}",
            f"{item['risk_level']}",
            f"{item['confidence_percent']}%"
        ])
    top3_table = Table(top3_rows, colWidths=[45, 265, 130, 100])
    top3_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1E3A8A')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8.5),
        ('PADDING', (0, 0), (-1, -1), 4),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F8FAFC')])
    ]))
    story.append(top3_table)
    story.append(Spacer(1, 8))
    
    # 6. Complete 7-Class Confidence Distribution
    story.append(Paragraph("<b>Seven-Class Confidence Distribution</b>", section_heading))
    dist_rows = [["Code", "Condition", "Risk Category", "Score %"]]
    for item in dist:
        dist_rows.append([
            item['code'].upper(),
            item['short_name'],
            item['risk_level'],
            f"{item['confidence_percent']}%"
        ])
    dist_table = Table(dist_rows, colWidths=[55, 255, 130, 100])
    dist_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#334155')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('PADDING', (0, 0), (-1, -1), 3),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#E2E8F0')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F8FAFC')])
    ]))
    story.append(dist_table)
    story.append(Spacer(1, 10))
    
    # 7. Medical Disclaimer Callout Box
    disclaimer_box = Table([
        [
            Paragraph("<b>IMPORTANT MEDICAL DISCLAIMER:</b>", ParagraphStyle('DisTitle', parent=disclaimer_style, fontName='Helvetica-Bold', textColor=colors.HexColor('#9A3412'))),
        ],
        [
            Paragraph(MEDICAL_DISCLAIMER, disclaimer_style)
        ]
    ], colWidths=[540])
    disclaimer_box.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#FEF3C7')),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#F59E0B')),
        ('PADDING', (0, 0), (-1, -1), 6)
    ]))
    story.append(KeepTogether([disclaimer_box]))
    
    # Build Document
    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes
