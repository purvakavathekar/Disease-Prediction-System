"""
PDF Assessment Report Generator Module
Constructs a formatted, downloadable clinical assessment summary PDF using ReportLab.
Uses saved assessment results directly without triggering any new predictions.
"""
import io
import json
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, KeepTogether
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
import explainability

def generate_assessment_pdf(assessment_data):
    """
    Generates a PDF document for the given saved assessment dictionary.
    Returns bytes of the generated PDF.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()

    # Custom styles
    brand_style = ParagraphStyle(
        'BrandTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#2b7a97')
    )

    subtitle_style = ParagraphStyle(
        'BrandSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=13,
        textColor=colors.HexColor('#64748b')
    )

    h2_style = ParagraphStyle(
        'SectionHeading',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor('#1e293b'),
        spaceBefore=8,
        spaceAfter=4
    )

    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor('#334155')
    )

    badge_style = ParagraphStyle(
        'DiseaseBadge',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=18,
        textColor=colors.HexColor('#0369a1')
    )

    warning_style = ParagraphStyle(
        'WarningBoxText',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor('#92400e')
    )

    danger_style = ParagraphStyle(
        'DangerBoxText',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor('#991b1b')
    )

    disclaimer_style = ParagraphStyle(
        'DisclaimerText',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=8,
        leading=11,
        textColor=colors.HexColor('#92400e')
    )

    story = []

    # 1. Header Banner
    asmt_id = assessment_data.get('id', 'N/A')
    timestamp = assessment_data.get('timestamp', 'N/A')
    patient_name = assessment_data.get('patient_name', 'Anonymous')
    patient_age = assessment_data.get('patient_age', 'N/A')
    patient_gender = assessment_data.get('patient_gender', 'Not Specified')
    predicted_disease = assessment_data.get('predicted_disease', 'Undetermined')
    symptoms = assessment_data.get('symptoms', [])
    follow_up = assessment_data.get('follow_up', {})
    symptom_vector = assessment_data.get('symptom_vector', '')

    header_data = [
        [
            Paragraph("<b>HealthDiagnose</b> | Clinical Assessment Report", brand_style),
            Paragraph(f"<b>Record ID:</b> {asmt_id}<br/><b>Date:</b> {timestamp}", subtitle_style)
        ]
    ]
    t_header = Table(header_data, colWidths=[360, 180])
    t_header.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('ALIGN', (1, 0), (1, 0), 'RIGHT'),
    ]))
    story.append(t_header)
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#4aa8c5'), spaceBefore=8, spaceAfter=12))

    # Red-Flag Emergency Banner if critical symptoms (chest pain, breathlessness, etc.)
    red_flag_note = follow_up.get('red_flag_note')
    if not red_flag_note:
        clean_symps = [str(s).lower().replace(' ', '_') for s in symptoms]
        if any(rf in clean_symps for rf in ['chest_pain', 'breathlessness', 'blood_in_sputum', 'stomach_bleeding']):
            red_flag_note = "EMERGENCY RED-FLAG: Critical symptom(s) reported (Chest pain / Breathlessness). Seek immediate emergency medical care if experiencing acute severe symptoms."

    if red_flag_note:
        t_danger = Table([[Paragraph(f"<b>RED-FLAG ALERT:</b> {red_flag_note}", danger_style)]], colWidths=[540])
        t_danger.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#fee2e2')),
            ('BOX', (0, 0), (-1, -1), 1.2, colors.HexColor('#ef4444')),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ]))
        story.append(t_danger)
        story.append(Spacer(1, 8))

    # 2. Patient Demographics & Assessment Overview Table
    demo_data = [
        [
            Paragraph("<b>Patient Name:</b>", body_style), Paragraph(str(patient_name), body_style),
            Paragraph("<b>Assessment ID:</b>", body_style), Paragraph(str(asmt_id), body_style)
        ],
        [
            Paragraph("<b>Age:</b>", body_style), Paragraph(str(patient_age or 'N/A'), body_style),
            Paragraph("<b>Date / Time:</b>", body_style), Paragraph(str(timestamp), body_style)
        ],
        [
            Paragraph("<b>Gender:</b>", body_style), Paragraph(str(patient_gender or 'N/A'), body_style),
            Paragraph("<b>Evaluation Type:</b>", body_style), Paragraph("132-Symptom ML Model", body_style)
        ]
    ]
    t_demo = Table(demo_data, colWidths=[90, 180, 100, 170])
    t_demo.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f8fafc')),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#f1f5f9')),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(t_demo)
    story.append(Spacer(1, 10))

    # 3. Model Diagnostic Outcome & Top 3 Differential Diagnoses
    story.append(Paragraph("Diagnostic Model Prediction & Top Differential Diagnoses", h2_style))
    top_3_diseases = follow_up.get('top_3_diseases') or []
    if top_3_diseases:
        top_diff_str = "<br/>".join([
            f"<b>{i+1}. {d['disease']}</b>: {d['probability']}% probability"
            for i, d in enumerate(top_3_diseases[:3])
        ])
    else:
        top_diff_str = f"<b>Primary Candidate:</b> {predicted_disease}"

    pred_data = [
        [
            Paragraph(f"<b>Primary Outcome:</b><br/>{predicted_disease}", badge_style),
            Paragraph(f"<b>Top Ranked Conditions:</b><br/>{top_diff_str}", body_style)
        ]
    ]
    t_pred = Table(pred_data, colWidths=[270, 270])
    t_pred.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#e0f2fe')),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#bae6fd')),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(t_pred)

    # Low Confidence Advisory Warning (<60%)
    low_conf_warn = follow_up.get('low_confidence_warning')
    if low_conf_warn:
        story.append(Spacer(1, 6))
        t_warn = Table([[Paragraph(f"<b>Low Confidence Advisory:</b> {low_conf_warn}", warning_style)]], colWidths=[540])
        t_warn.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#fef3c7')),
            ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#f59e0b')),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ]))
        story.append(t_warn)

    story.append(Spacer(1, 10))

    # 4. Reported Symptoms (with individual severities if provided)
    story.append(Paragraph(f"Reported Symptoms ({len(symptoms)} Total)", h2_style))
    symp_severities = follow_up.get('symptom_severities', {}) if isinstance(follow_up, dict) else {}
    SEV_LABEL_MAP = {1: 'Very Mild', 2: 'Mild', 3: 'Moderate', 4: 'Severe', 5: 'Very Severe'}
    
    formatted_symps = []
    for s in symptoms:
        s_sev = symp_severities.get(s) or symp_severities.get(s.strip().lower().replace(' ', '_'))
        if s_sev is not None:
            try:
                num = int(str(s_sev).split('-')[0].strip())
                label = SEV_LABEL_MAP.get(num, f"Severity {num}/5")
            except Exception:
                label = str(s_sev)
            formatted_symps.append(f"<b>{s}</b> [{label}]")
        else:
            formatted_symps.append(s)

    symp_text = ", ".join(formatted_symps) if formatted_symps else "None recorded"
    story.append(Paragraph(symp_text, body_style))
    story.append(Spacer(1, 10))

    # 5. About This Prediction (Educational & Pattern-Matching Context)
    xai_data = None
    if symptom_vector:
        try:
            xai_data = explainability.explain_prediction(symptom_vector, predicted_disease=predicted_disease)
        except Exception:
            xai_data = None

    story.append(Paragraph("About This Prediction", h2_style))
    story.append(Paragraph(f"<b>Predicted Condition:</b> {predicted_disease}", body_style))
    story.append(Spacer(1, 4))
    story.append(Paragraph(f"<b>Selected Symptoms:</b> {symp_text}", body_style))
    story.append(Spacer(1, 4))
    explanation_txt = (xai_data.get('explanation') if xai_data else
                       "The model generated this prediction by evaluating the submitted symptom pattern against patterns in its training data.")
    story.append(Paragraph(f"<b>Explanation:</b> {explanation_txt}", body_style))
    story.append(Spacer(1, 4))

    if xai_data and xai_data.get('alternative_conditions'):
        alts = ", ".join(xai_data['alternative_conditions'])
        story.append(Paragraph(f"<b>Alternative Pattern Matches:</b> {alts}", body_style))
        story.append(Spacer(1, 4))

    disclaimer_p = Paragraph(
        "<i>Disclaimer: This assessment is for informational and educational purposes only. "
        "It cannot confirm or rule out a medical condition. Always consult a qualified healthcare professional.</i>",
        body_style
    )
    story.append(disclaimer_p)
    story.append(Spacer(1, 10))

    # 6. Contextual Follow-Up & Daily Health Assessment Findings
    if follow_up and isinstance(follow_up, dict) and len(follow_up) > 0:
        if follow_up.get('is_daily_assessment'):
            story.append(Paragraph("Daily Health Assessment & Progression Tracking", h2_style))
            trend = follow_up.get('overall_trend', 'Stable / Unchanged')
            story.append(Paragraph(f"<b>Overall Progression Trend:</b> {trend}", body_style))
            story.append(Spacer(1, 4))

            daily_table_data = [
                [Paragraph("<b>Category</b>", body_style), Paragraph("<b>Symptoms & Severity Details</b>", body_style)]
            ]

            # Improving
            imp = follow_up.get('improving_symptoms', [])
            if imp:
                imp_text = ", ".join([f"{s['symptom_name']} (Severity: {s.get('previous_severity','?')}/5 &rarr; {s.get('current_severity','?')}/5)" for s in imp])
                daily_table_data.append([Paragraph("<b>Improving Symptoms:</b>", body_style), Paragraph(imp_text, body_style)])

            # Worsening
            wors = follow_up.get('worsening_symptoms', [])
            if wors:
                wors_text = ", ".join([f"{s['symptom_name']} (Severity: {s.get('previous_severity','?')}/5 &rarr; {s.get('current_severity','?')}/5)" for s in wors])
                daily_table_data.append([Paragraph("<b>Worsening Symptoms:</b>", body_style), Paragraph(wors_text, body_style)])

            # Unchanged
            unch = follow_up.get('unchanged_symptoms', [])
            if unch:
                unch_text = ", ".join([f"{s['symptom_name']} (Severity: {s.get('current_severity','?')}/5)" for s in unch])
                daily_table_data.append([Paragraph("<b>Unchanged Symptoms:</b>", body_style), Paragraph(unch_text, body_style)])

            # Resolved
            res = follow_up.get('resolved_symptoms', [])
            if res:
                res_text = ", ".join([f"{s['symptom_name']} (Previously: {s.get('previous_severity','?')}/5)" for s in res])
                daily_table_data.append([Paragraph("<b>Resolved Symptoms:</b>", body_style), Paragraph(res_text, body_style)])

            # New
            new_symps = follow_up.get('new_symptoms', [])
            if new_symps:
                new_text = ", ".join([f"{s['symptom_name']} (Severity: {s.get('current_severity','?')}/5)" for s in new_symps])
                daily_table_data.append([Paragraph("<b>Newly Reported:</b>", body_style), Paragraph(new_text, body_style)])

            # Notes
            notes = follow_up.get('notes')
            if notes:
                daily_table_data.append([Paragraph("<b>Patient Notes:</b>", body_style), Paragraph(str(notes), body_style)])

            t_daily = Table(daily_table_data, colWidths=[150, 390])
            t_daily.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f1f5f9')),
                ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
                ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
                ('TOPPADDING', (0, 0), (-1, -1), 4),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ]))
            story.append(t_daily)
            story.append(Spacer(1, 10))

        else:
            story.append(Paragraph("Contextual Questionnaire Findings", h2_style))
            fu_table_data = [
                [Paragraph("<b>Question Context</b>", body_style), Paragraph("<b>Recorded Patient Response</b>", body_style)]
            ]
            for k, v in follow_up.items():
                if v and k not in ('is_daily_assessment',):
                    label = k.replace('_', ' ').title()
                    fu_table_data.append([
                        Paragraph(label, body_style),
                        Paragraph(str(v), body_style)
                    ])

            t_fu = Table(fu_table_data, colWidths=[180, 360])
            t_fu.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f1f5f9')),
                ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
                ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
                ('TOPPADDING', (0, 0), (-1, -1), 4),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ]))
            story.append(t_fu)
            story.append(Spacer(1, 12))

    # 7. Official Medical Disclaimer
    disclaimer_box = [
        [
            Paragraph(
                "<b>OFFICIAL NOTICE & DISCLAIMER:</b> This report is generated by a computer-based machine learning "
                "algorithm for educational and self-tracking purposes only. The findings contained herein do NOT "
                "substitute for clinical judgment, definitive diagnostic laboratory tests, or medical consultation with a licensed "
                "physician. In case of emergency or severe discomfort, contact medical services immediately.",
                disclaimer_style
            )
        ]
    ]
    t_disc = Table(disclaimer_box, colWidths=[540])
    t_disc.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#fff8e6')),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#f59e0b')),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ('RIGHTPADDING', (0, 0), (-1, -1), 10),
    ]))
    story.append(KeepTogether([t_disc]))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()
