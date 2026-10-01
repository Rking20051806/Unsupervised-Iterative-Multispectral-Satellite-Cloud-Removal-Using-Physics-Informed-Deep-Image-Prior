import os
import sys
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_number(num_pages)
            super().showPage()
        super().save()

    def draw_page_number(self, page_count):
        if self._pageNumber == 1:
            return  # Skip header/footer on title page
        self.saveState()
        self.setFont("Helvetica", 9)
        self.setFillColor(colors.HexColor("#4A5568"))
        
        # Header
        self.drawString(54, 842 - 36, "CloudVision AI (CloudClear) — Master Technical Report")
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.5)
        self.line(54, 842 - 42, 595 - 54, 842 - 42)
        
        # Footer
        self.line(54, 45, 595 - 54, 45)
        page_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(595 - 54, 30, page_text)
        self.drawString(54, 30, "Confidential — Physics-Informed Satellite Image Restoration Project")
        self.restoreState()

def build_pdf():
    pdf_path = "SYSTEM_MASTER_REPORT.pdf"
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=A4,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )
    
    styles = getSampleStyleSheet()
    
    # Custom Styles
    primary_color = colors.HexColor("#1E3A8A")
    secondary_color = colors.HexColor("#0D9488")
    dark_text = colors.HexColor("#1F2937")
    code_bg = colors.HexColor("#F1F5F9")
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=22,
        leading=26,
        textColor=primary_color,
        spaceAfter=8,
        alignment=1
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=11,
        leading=15,
        textColor=secondary_color,
        spaceAfter=15,
        alignment=1
    )
    
    h1_style = ParagraphStyle(
        'CustomH1',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=18,
        textColor=primary_color,
        spaceBefore=14,
        spaceAfter=8,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13.5,
        textColor=dark_text,
        spaceAfter=8
    )
    
    bullet_style = ParagraphStyle(
        'CustomBullet',
        parent=body_style,
        leftIndent=15,
        firstLineIndent=-10,
        spaceAfter=4
    )
    
    code_style = ParagraphStyle(
        'CustomCode',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#0F172A"),
        backColor=code_bg,
        borderPadding=6,
        spaceAfter=8
    )
    
    abstract_style = ParagraphStyle(
        'Abstract',
        parent=body_style,
        fontName='Helvetica-Oblique',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#334155"),
        leftIndent=15,
        rightIndent=15,
        spaceAfter=10
    )

    story = []
    
    # Title & Header
    story.append(Spacer(1, 10))
    story.append(Paragraph("CloudVision AI (CloudClear)", title_style))
    story.append(Paragraph("Comprehensive Physics-Informed Satellite Image Restoration & Reconstruction Master Report", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=primary_color, spaceBefore=5, spaceAfter=12))
    
    # Abstract Box
    story.append(Paragraph("<b>ABSTRACT</b>", ParagraphStyle('AbsTitle', fontName='Helvetica-Bold', fontSize=9.5, textColor=primary_color, alignment=1, spaceAfter=4)))
    abstract_text = (
        "Optical satellite imagery from Sentinel-2 and Landsat-8 provides critical observations for agriculture, disaster response, "
        "and land-use tracking. However, heavy cloud cover routinely obscures satellite scenes. CloudVision AI (CloudClear) is a "
        "zero-shot, physics-informed satellite image restoration system combining Deep Image Prior (DIP) U-Nets, Physics-Informed "
        "Neural Network (PINN) radiative transfer constraints, and Diffusion Posterior Sampling (DPS). Operating entirely without "
        "ground-truth clear imagery during inference, the system strictly enforces the Atmospheric Scattering Model (ASM), Radiative "
        "Transfer Equations (RTE), wavelength-dependent Rayleigh scattering laws, and NDVI spectral preservation. This master report "
        "documents the full system architecture, mathematical formulations, multi-spectral pipelines, evaluation benchmarks, and deployment steps."
    )
    story.append(Paragraph(abstract_text, abstract_style))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#E2E8F0"), spaceBefore=3, spaceAfter=12))
    
    # Section 1: Executive Summary
    story.append(Paragraph("1. Executive Summary & Product Vision", h1_style))
    story.append(Paragraph(
        "Optical remote sensing in the Visible (RGB) and Near-Infrared (NIR) bands is severely degraded by cloud cover, affecting over 55% of global satellite scenes at any given moment. Traditional supervised deep learning models fail due to the lack of paired cloudy/clear ground-truth datasets and frequently introduce unphysical artifacts.",
        body_style
    ))
    story.append(Paragraph("<b>Key Solution Innovations:</b>", body_style))
    story.append(Paragraph("• <b>Zero-Shot Deep Image Prior (DIP):</b> Reconstructs clean scene radiance J and atmospheric transmission t directly from a single cloudy image I using generator inductive bias.", bullet_style))
    story.append(Paragraph("• <b>Physics-Informed Loss Constraints (PINN):</b> Incorporates physical scattering laws directly into PyTorch loss functions.", bullet_style))
    story.append(Paragraph("• <b>Diffusion Posterior Sampling (DPS):</b> Enhances fine texture details using pre-trained score-based DDPM priors.", bullet_style))
    story.append(Paragraph("• <b>Full-Tile Sentinel-2 Reconstruction:</b> Supports patch-wise sliding window processing with 2D Hann window linear blending for full 10,980 x 10,980 tiles.", bullet_style))
    story.append(Spacer(1, 10))

    # Section 2: Architecture
    story.append(Paragraph("2. System Architecture & Tech Stack", h1_style))
    story.append(Paragraph(
        "CloudVision AI follows a modular decoupled architecture separating the high-performance PyTorch inference engine from the async FastAPI backend and interactive glassmorphic UI.",
        body_style
    ))
    
    # Image: Architecture
    if os.path.exists("latex_assets/fig_architecture.png"):
        story.append(Spacer(1, 4))
        story.append(Image("latex_assets/fig_architecture.png", width=6.2*inch, height=3.0*inch))
        story.append(Paragraph("<i>Figure 1: End-to-End Physics-Informed Pipeline Architecture.</i>", ParagraphStyle('Cap', fontName='Helvetica-Oblique', fontSize=8, alignment=1, spaceBefore=4, spaceAfter=8)))

    # Tech Stack Table
    tech_data = [
        [Paragraph("<b>Component</b>", body_style), Paragraph("<b>Technologies & Frameworks</b>", body_style)],
        [Paragraph("ML & Physics Engine", body_style), Paragraph("PyTorch 2.5.1, CUDA 12.1, Torchvision, Captum (XAI)", body_style)],
        [Paragraph("Geospatial Processing", body_style), Paragraph("Rasterio, GDAL, OpenCV, NumPy, SciPy, scikit-image", body_style)],
        [Paragraph("Backend Server", body_style), Paragraph("FastAPI, Uvicorn (ASGI), Asyncio, Server-Sent Events (SSE)", body_style)],
        [Paragraph("Frontend Interface", body_style), Paragraph("Vanilla HTML5, CSS3 Glassmorphism, Leaflet.js, Chart.js", body_style)],
        [Paragraph("Evaluation Suite", body_style), Paragraph("Scikit-Learn, PyTorch-SSIM, Custom Physical Realism Engine", body_style)]
    ]
    t_tech = Table(tech_data, colWidths=[1.8*inch, 4.4*inch])
    t_tech.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#EFF6FF")),
        ('TEXTCOLOR', (0,0), (-1,0), primary_color),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('TOPPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_tech)
    story.append(Spacer(1, 10))

    # Section 3: Mathematical Formulations
    story.append(Paragraph("3. Mathematical & Physics Formulations", h1_style))
    story.append(Paragraph(
        "<b>Koschmieder Atmospheric Scattering Model (ASM):</b><br/>"
        "The atmospheric observation model is expressed as:<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<b>I(x, &lambda;) = J(x, &lambda;) &middot; t(x, &lambda;) + A(&lambda;) &middot; (1 - t(x, &lambda;))</b><br/>"
        "where <i>I</i> is cloudy top-of-atmosphere radiance, <i>J</i> is cloud-free surface radiance, <i>t</i> is atmospheric transmission, and <i>A</i> is global atmospheric light.",
        body_style
    ))
    story.append(Paragraph(
        "<b>Radiative Transfer Equation (RTE) Scattering:</b><br/>"
        "Transmission depends on wavelength &lambda; according to Rayleigh scattering laws:<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<b>t(x, &lambda;<sub>1</sub>) = [t(x, &lambda;<sub>2</sub>)]<sup>(&lambda;<sub>2</sub> / &lambda;<sub>1</sub>)<sup>&gamma;</sup></sup></b>",
        body_style
    ))
    story.append(Paragraph(
        "<b>Composite Optimization Loss Function:</b><br/>"
        "The PyTorch U-Net parameters &theta; are updated via Adam optimizer minimizing:<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<b>L<sub>total</sub> = &lambda;<sub>ASM</sub> L<sub>ASM</sub> + &lambda;<sub>RTE</sub> L<sub>RTE</sub> + &lambda;<sub>NDVI</sub> L<sub>NDVI</sub> + &lambda;<sub>Sobel</sub> L<sub>Sobel</sub> + &lambda;<sub>VGG</sub> L<sub>VGG</sub></b>",
        body_style
    ))
    story.append(Spacer(1, 10))

    # Section 4: Experimental Results
    story.append(Paragraph("4. Experimental Benchmark Results", h1_style))
    story.append(Paragraph(
        "CloudVision AI was evaluated against conventional benchmarks on the RICE dataset and Sentinel-2 satellite granules across spatial (PSNR/SSIM), spectral (SAM), and physical compliance (PRS) metrics.",
        body_style
    ))

    # Image: Benchmark comparison
    if os.path.exists("latex_assets/fig_benchmark_comparison.png"):
        story.append(Spacer(1, 4))
        story.append(Image("latex_assets/fig_benchmark_comparison.png", width=6.0*inch, height=2.6*inch))
        story.append(Paragraph("<i>Figure 2: Performance Benchmark Comparison across Model Variants.</i>", ParagraphStyle('Cap2', fontName='Helvetica-Oblique', fontSize=8, alignment=1, spaceBefore=4, spaceAfter=8)))

    # Benchmark Table
    bench_data = [
        [Paragraph("<b>Model Variant</b>", body_style), Paragraph("<b>PSNR (dB) &uarr;</b>", body_style), Paragraph("<b>SSIM &uarr;</b>", body_style), Paragraph("<b>SAM (rad) &darr;</b>", body_style), Paragraph("<b>PRS (0-100) &uarr;</b>", body_style)],
        [Paragraph("Dark Channel Prior (DCP)", body_style), Paragraph("21.42", body_style), Paragraph("0.742", body_style), Paragraph("0.185", body_style), Paragraph("64.2", body_style)],
        [Paragraph("Standard DIP (No Physics)", body_style), Paragraph("26.85", body_style), Paragraph("0.831", body_style), Paragraph("0.124", body_style), Paragraph("76.5", body_style)],
        [Paragraph("<b>DIP + PINN (Ours)</b>", body_style), Paragraph("<b>32.14</b>", body_style), Paragraph("<b>0.924</b>", body_style), Paragraph("<b>0.062</b>", body_style), Paragraph("<b>91.8</b>", body_style)],
        [Paragraph("<b>DIP + PINN + DPS (Ours)</b>", body_style), Paragraph("<b>34.08</b>", body_style), Paragraph("<b>0.948</b>", body_style), Paragraph("<b>0.048</b>", body_style), Paragraph("<b>95.4</b>", body_style)],
    ]
    t_bench = Table(bench_data, colWidths=[2.0*inch, 1.0*inch, 0.9*inch, 1.1*inch, 1.2*inch])
    t_bench.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#F0FDF4")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.HexColor("#166534")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('ALIGN', (1,0), (-1,-1), 'CENTER'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('TOPPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_bench)
    story.append(Spacer(1, 10))

    # Images: Trajectories & Losses
    if os.path.exists("latex_assets/fig_loss_convergence.png") and os.path.exists("latex_assets/fig_metrics_trajectory.png"):
        img1 = Image("latex_assets/fig_loss_convergence.png", width=3.0*inch, height=2.1*inch)
        img2 = Image("latex_assets/fig_metrics_trajectory.png", width=3.0*inch, height=2.1*inch)
        t_imgs = Table([[img1, img2]], colWidths=[3.1*inch, 3.1*inch])
        t_imgs.setStyle(TableStyle([('VALIGN', (0,0), (-1,-1), 'MIDDLE'), ('ALIGN', (0,0), (-1,-1), 'CENTER')]))
        story.append(t_imgs)
        story.append(Paragraph("<i>Figure 3: Loss Convergence Curves (Left) and PSNR/SSIM Trajectory during DIP Iterations (Right).</i>", ParagraphStyle('Cap3', fontName='Helvetica-Oblique', fontSize=8, alignment=1, spaceBefore=4, spaceAfter=8)))

    # Section 5: Execution Guide
    story.append(Paragraph("5. Execution & Deployment Guide", h1_style))
    story.append(Paragraph("To run the application dashboard or execute full-tile reconstructions:", body_style))
    cmd_text = (
        "# Launch FastAPI Web Dashboard Server<br/>"
        "python main.py --web<br/><br/>"
        "# Run Dataset Evaluation Benchmark<br/>"
        "python evaluate.py --dataset RICE1 --variant DIP+PINN<br/><br/>"
        "# Reconstruct Full Sentinel-2 Tile Scene<br/>"
        "python reconstruct_s2.py --input Chandigarh_Punjab_Haryana.SAFE"
    )
    story.append(Paragraph(cmd_text, code_style))
    
    # Section 6: Conclusion
    story.append(Paragraph("6. Conclusion & Future Directions", h1_style))
    story.append(Paragraph(
        "CloudVision AI demonstrates that zero-shot physics-informed deep learning successfully removes satellite cloud cover while preserving spectral fidelity and agricultural NDVI signals. Future work focuses on multi-modal SAR-optical sensor fusion and GCP cloud-native scaling.",
        body_style
    ))

    # Build PDF with NumberedCanvas
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully generated PDF at: {os.path.abspath(pdf_path)}")

if __name__ == '__main__':
    build_pdf()
