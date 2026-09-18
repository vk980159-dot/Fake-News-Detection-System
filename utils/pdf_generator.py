import io
from datetime import datetime
from typing import Dict, Any, List
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable

def generate_pdf_report(report_data: Dict[str, Any]) -> bytes:
    """
    Generate an enhanced, professional PDF verification report containing:
    - Input Type & Title/Claim
    - Extracted Text Summary
    - ML Prediction & Model Confidence
    - Final Verdict & Evidence Status
    - Source Information
    - Supporting & Contradicting Evidence
    - AI Reasoning
    - Timestamp & Methodology Note
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )
    
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#1E293B'),
        spaceAfter=6
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#64748B'),
        spaceAfter=15
    )
    
    section_heading = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=17,
        textColor=colors.HexColor('#0F172A'),
        spaceBefore=12,
        spaceAfter=6
    )
    
    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor('#334155')
    )
    
    verdict_colors = {
        "VERIFIED": colors.HexColor('#16A34A'),
        "MISLEADING": colors.HexColor('#EA580C'),
        "LIKELY FAKE": colors.HexColor('#DC2626'),
        "UNCERTAIN": colors.HexColor('#64748B')
    }
    
    verdict = report_data.get("verdict", "UNCERTAIN")
    verdict_color = verdict_colors.get(verdict, colors.HexColor('#2563EB'))
    
    elements = []
    
    # ── Header ────────────────────────────────────────────────────────────
    elements.append(Paragraph("🌐 TruthLens AI", title_style))
    elements.append(Paragraph(
        "AI-Powered Global News Verification System &nbsp;|&nbsp; <i>\"Don't trust the headline. Verify the claim.\"</i>",
        subtitle_style
    ))
    elements.append(Paragraph(
        f"Automated Fact-Checking Dossier &nbsp;|&nbsp; Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        ParagraphStyle('SubSub', parent=subtitle_style, fontSize=8, textColor=colors.HexColor('#94A3B8'), spaceAfter=10)
    ))
    elements.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#CBD5E1'), spaceAfter=12))
    
    # ── Executive Verdict Box ──────────────────────────────────────────────
    verdict_style = ParagraphStyle(
        'VerdictText',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=verdict_color,
        alignment=1 # Centered
    )
    
    verdict_table_data = [
        [Paragraph(f"FINAL VERDICT: {verdict}", verdict_style)],
        [Paragraph(f"<b>Model Confidence:</b> {report_data.get('confidence', 0):.2f}% &nbsp;|&nbsp; <b>ML Prediction:</b> {report_data.get('ml_prediction', 'N/A')} &nbsp;|&nbsp; <b>Evidence Status:</b> {report_data.get('evidence_status', 'N/A')}", ParagraphStyle('VerdictSub', parent=body_style, alignment=1))]
    ]
    verdict_table = Table(verdict_table_data, colWidths=[530])
    verdict_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F8FAFC')),
        ('BOX', (0, 0), (-1, -1), 1.5, verdict_color),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('LEFTPADDING', (0, 0), (-1, -1), 12),
        ('RIGHTPADDING', (0, 0), (-1, -1), 12),
    ]))
    elements.append(verdict_table)
    elements.append(Spacer(1, 10))
    
    # ── Input & Article Summary ───────────────────────────────────────────
    elements.append(Paragraph("📌 Content Information", section_heading))
    input_info = [
        [Paragraph("<b>Input Type:</b>", body_style), Paragraph(str(report_data.get("input_type", "Text")), body_style)],
        [Paragraph("<b>Title / Claim:</b>", body_style), Paragraph(str(report_data.get("title", "N/A"))[:150], body_style)],
        [Paragraph("<b>Source / Domain:</b>", body_style), Paragraph(str(report_data.get("domain", "Direct Input")), body_style)],
        [Paragraph("<b>Author / Date:</b>", body_style), Paragraph(f"{report_data.get('author', 'Not Available')} / {report_data.get('publish_date', 'Not Available')}", body_style)],
    ]
    info_table = Table(input_info, colWidths=[130, 400])
    info_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F1F5F9')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#E2E8F0')),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    elements.append(info_table)
    elements.append(Spacer(1, 8))
    
    # ── Text Summary Excerpt ───────────────────────────────────────────────
    raw_text = report_data.get("text_summary", "")
    if raw_text:
        excerpt = (raw_text[:350] + "...") if len(raw_text) > 350 else raw_text
        elements.append(Paragraph("<b>Extracted Content Excerpt:</b>", body_style))
        elements.append(Paragraph(f"<i>\"{excerpt}\"</i>", ParagraphStyle('Excerpt', parent=body_style, textColor=colors.HexColor('#475569'))))
        elements.append(Spacer(1, 8))
    
    # ── AI Reasoning & Evidence Fusion ─────────────────────────────────────
    elements.append(Paragraph("🤖 AI Reasoning & Decision Rationale", section_heading))
    reasoning_list = report_data.get("reasoning", [])
    if isinstance(reasoning_list, list) and reasoning_list:
        for r in reasoning_list:
            elements.append(Paragraph(f"• {r}", body_style))
            elements.append(Spacer(1, 3))
    elif isinstance(reasoning_list, str):
        elements.append(Paragraph(reasoning_list, body_style))
    else:
        elements.append(Paragraph("No automated reasoning generated.", body_style))
    elements.append(Spacer(1, 8))
    
    # ── Evidence Breakdown ────────────────────────────────────────────────
    elements.append(Paragraph("🔍 Evidence Corroboration", section_heading))
    supporting = report_data.get("supporting_evidence", [])
    contradicting = report_data.get("contradicting_evidence", [])
    
    if supporting:
        elements.append(Paragraph("<b>Supporting Sources Found:</b>", ParagraphStyle('SuppHead', parent=body_style, textColor=colors.HexColor('#15803D'))))
        for item in supporting[:3]:
            title = item.get('title', 'Untitled')
            source = item.get('source', 'Unknown Source')
            link = item.get('url', '')
            tier = item.get('source_tier') or item.get('tier_badge') or 'Public Source'
            tier_str = f" [{tier}]" if tier else ""
            elements.append(Paragraph(f"• <b>[{source}]{tier_str}</b> {title} ({link[:55]}...)" if len(link) > 55 else f"• <b>[{source}]{tier_str}</b> {title} ({link})", body_style))
            elements.append(Spacer(1, 2))
        elements.append(Spacer(1, 4))
    else:
        elements.append(Paragraph("• <i>No direct corroborating public reports identified.</i>", body_style))
        elements.append(Spacer(1, 4))
        
    if contradicting:
        elements.append(Paragraph("<b>Contradicting / Debunking Sources Found:</b>", ParagraphStyle('ContHead', parent=body_style, textColor=colors.HexColor('#B91C1C'))))
        for item in contradicting[:3]:
            title = item.get('title', 'Untitled')
            source = item.get('source', 'Unknown Source')
            link = item.get('url', '')
            tier = item.get('source_tier') or item.get('tier_badge') or 'Public Source'
            tier_str = f" [{tier}]" if tier else ""
            elements.append(Paragraph(f"• <b>[{source}]{tier_str}</b> {title} ({link[:55]}...)" if len(link) > 55 else f"• <b>[{source}]{tier_str}</b> {title} ({link})", body_style))
            conflict_type = item.get('conflict_type')
            if conflict_type:
                claim_attr = item.get('claim_attribute', '')
                ev_attr = item.get('evidence_attribute', '')
                conflict_detail = f"&nbsp;&nbsp;&nbsp;&nbsp;🚨 <i>{conflict_type}: Claim asserted '{claim_attr}' vs Evidence reported '{ev_attr}'</i>"
                elements.append(Paragraph(conflict_detail, ParagraphStyle('ConflictStyle', parent=body_style, textColor=colors.HexColor('#DC2626'), fontSize=8.5)))
            elements.append(Spacer(1, 2))
        elements.append(Spacer(1, 4))
    else:
        elements.append(Paragraph("• <i>No direct contradictory reports identified.</i>", body_style))
        elements.append(Spacer(1, 4))
        
    # ── Technical Disclaimer & Viva Information ────────────────────────────
    elements.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#E2E8F0'), spaceBefore=10, spaceAfter=8))
    disclaimer_text = (
        "<b>Responsible AI Disclaimer:</b> TruthLens AI provides probabilistic factual assessment combining "
        "statistical NLP with live public corroborating evidence. This report does not constitute absolute legal or "
        "journalistic proof. Verification should be independently confirmed for critical decision-making. "
        "Model confidence indicates classifier certainty, not factual truth."
    )
    elements.append(Paragraph(disclaimer_text, ParagraphStyle('Disclaimer', parent=styles['Italic'], fontSize=8, leading=11, textColor=colors.HexColor('#94A3B8'))))
    
    doc.build(elements)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes
