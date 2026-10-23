import os
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import qn, nsdecls
import win32com.client

REPORT_DIR = r"D:\COLLAGE CLASSES\SEM - 03\Machine Learning\Project\Project Report"
TEMPLATE_PATH = os.path.join(REPORT_DIR, "Dijkstra_Project_Report.docx")
OUTPUT_DOCX = os.path.join(REPORT_DIR, "Wine_Quality_Prediction_Project_Report.docx")
OUTPUT_PDF = os.path.join(REPORT_DIR, "Wine_Quality_Prediction_Project_Report.pdf")

SCREENSHOTS_DIR = os.path.join(REPORT_DIR, "screenshots")
REPORTS_DIR = r"D:\COLLAGE CLASSES\SEM - 03\Machine Learning\Project\Model\reports"

def set_run_font(run, name="Times New Roman", size_pt=9.5, bold=False, italic=False, color_rgb=None):
    run.font.name = name
    run.font.size = Pt(size_pt)
    run.bold = bold
    run.italic = italic
    if color_rgb:
        run.font.color.rgb = color_rgb
    rPr = run._element.get_or_add_rPr()
    rFonts = rPr.find(qn('w:rFonts'))
    if rFonts is None:
        rFonts = OxmlElement('w:rFonts')
        rPr.append(rFonts)
    rFonts.set(qn('w:ascii'), name)
    rFonts.set(qn('w:hAnsi'), name)
    rFonts.set(qn('w:cs'), name)

def add_para(doc, text="", align=WD_ALIGN_PARAGRAPH.JUSTIFY, size_pt=9.5, bold=False, italic=False, 
             space_before=0, space_after=3.0, line_spacing=1.12, color_rgb=None):
    p = doc.add_paragraph()
    p.alignment = align
    pf = p.paragraph_format
    pf.space_before = Pt(space_before)
    pf.space_after = Pt(space_after)
    pf.line_spacing = line_spacing
    if text:
        r = p.add_run(text)
        set_run_font(r, size_pt=size_pt, bold=bold, italic=italic, color_rgb=color_rgb)
    return p

def add_chapter_title(doc, text):
    return add_para(doc, text, align=WD_ALIGN_PARAGRAPH.CENTER, size_pt=14.0, bold=True, 
                    space_before=2, space_after=4, line_spacing=1.0)

def add_section_title(doc, text):
    return add_para(doc, text, align=WD_ALIGN_PARAGRAPH.LEFT, size_pt=11.0, bold=True, 
                    space_before=4, space_after=1.5, line_spacing=1.0)

def add_bullet(doc, bold_prefix, text, space_after=1.5):
    p = doc.add_paragraph(style='List Paragraph')
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    pf = p.paragraph_format
    pf.space_before = Pt(0)
    pf.space_after = Pt(space_after)
    pf.line_spacing = 1.10
    if bold_prefix:
        r1 = p.add_run(bold_prefix)
        set_run_font(r1, size_pt=9.0, bold=True)
    r2 = p.add_run(text)
    set_run_font(r2, size_pt=9.0, bold=False)
    return p

def add_page_break(doc):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(0)
    p.paragraph_format.line_spacing = 1.0
    r = p.add_run()
    r.add_break(docx.enum.text.WD_BREAK.PAGE)

def set_cell_shading(cell, color_hex):
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{color_hex}"/>')
    cell._tc.get_or_add_tcPr().append(shd)

def set_cell_margins(cell, top=60, bottom=60, left=90, right=90):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(f'''
        <w:tcMar {nsdecls("w")}>
            <w:top w:w="{top}" w:type="dxa"/>
            <w:bottom w:w="{bottom}" w:type="dxa"/>
            <w:left w:w="{left}" w:type="dxa"/>
            <w:right w:w="{right}" w:type="dxa"/>
        </w:tcMar>
    ''')
    tcPr.append(tcMar)

def set_table_borders(table, color="D3D3D3", sz="4", val="single"):
    tblPr = table._tbl.tblPr
    borders = parse_xml(f'''
        <w:tblBorders {nsdecls("w")}>
            <w:top w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>
            <w:bottom w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>
            <w:left w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>
            <w:right w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>
            <w:insideH w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>
            <w:insideV w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>
        </w:tblBorders>
    ''')
    tblPr.append(borders)

def add_framed_image(doc, img_path, width_inches, fig_label, caption_text):
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = tbl.cell(0, 0)
    set_cell_shading(cell, "FAFAFA")
    set_cell_margins(cell, top=30, bottom=30, left=30, right=30)
    tcPr = cell._tc.get_or_add_tcPr()
    borders = parse_xml(f'''
        <w:tcBorders {nsdecls("w")}>
            <w:top w:val="single" w:sz="4" w:space="0" w:color="CCCCCC"/>
            <w:bottom w:val="single" w:sz="4" w:space="0" w:color="CCCCCC"/>
            <w:left w:val="single" w:sz="4" w:space="0" w:color="CCCCCC"/>
            <w:right w:val="single" w:sz="4" w:space="0" w:color="CCCCCC"/>
        </w:tcBorders>
    ''')
    tcPr.append(borders)
    p = cell.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(1)
    p.paragraph_format.space_after = Pt(1)
    r = p.add_run()
    r.add_picture(img_path, width=Inches(width_inches))
    
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.space_before = Pt(1)
    cap.paragraph_format.space_after = Pt(2.5)
    r_fig = cap.add_run(fig_label + " — ")
    set_run_font(r_fig, size_pt=8.5, bold=True)
    r_text = cap.add_run(caption_text)
    set_run_font(r_text, size_pt=8.5, bold=False, italic=True)

def main():
    print("Loading base template:", TEMPLATE_PATH)
    doc = docx.Document(TEMPLATE_PATH)
    
    # Preserve sectPr at the end of body, clear all paragraphs and tables
    body = doc._body._element
    sectPr = doc.sections[0]._sectPr
    for child in list(body):
        if child.tag.endswith('sectPr'):
            continue
        body.remove(child)
        
    print("Constructing 10-page report...")
    
    # =========================================================================
    # PAGE 1: COVER PAGE
    # =========================================================================
    add_para(doc, "", space_before=10, space_after=6)
    add_para(doc, "Wine Quality Prediction System", align=WD_ALIGN_PARAGRAPH.CENTER, size_pt=24.0, bold=True, space_before=8, space_after=4)
    add_para(doc, "An End-to-End Machine Learning Web Application for Enological Sensory Assessment", align=WD_ALIGN_PARAGRAPH.CENTER, size_pt=13.0, italic=True, space_before=2, space_after=22)
    
    add_para(doc, "Project Report Submitted By", align=WD_ALIGN_PARAGRAPH.CENTER, size_pt=12.0, bold=False, space_before=6, space_after=3)
    add_para(doc, "Gopal Kumar", align=WD_ALIGN_PARAGRAPH.CENTER, size_pt=15.0, bold=True, space_before=2, space_after=3)
    add_para(doc, "UID : 25MCA20357", align=WD_ALIGN_PARAGRAPH.CENTER, size_pt=11.5, bold=False, space_before=1, space_after=2)
    add_para(doc, "Course / Branch: Master of Computer Applications (MCA)", align=WD_ALIGN_PARAGRAPH.CENTER, size_pt=11.5, bold=False, space_before=1, space_after=2)
    add_para(doc, "Subject: MACHINE LEARNING (MCA)", align=WD_ALIGN_PARAGRAPH.CENTER, size_pt=11.5, bold=False, space_before=1, space_after=22)
    
    add_para(doc, "Supervisor", align=WD_ALIGN_PARAGRAPH.CENTER, size_pt=12.0, bold=False, space_before=4, space_after=3)
    add_para(doc, "Mr. Sobit Rehan", align=WD_ALIGN_PARAGRAPH.CENTER, size_pt=13.5, bold=True, space_before=2, space_after=22)
    
    add_para(doc, "Chandigarh University", align=WD_ALIGN_PARAGRAPH.CENTER, size_pt=15.0, bold=True, space_before=6, space_after=3)
    add_para(doc, "Academic Year: 2025 – 2027", align=WD_ALIGN_PARAGRAPH.CENTER, size_pt=11.5, bold=False, space_before=1, space_after=2)
    add_para(doc, "Date of Submission: October 2026", align=WD_ALIGN_PARAGRAPH.CENTER, size_pt=11.5, bold=False, space_before=1, space_after=0)
    
    add_page_break(doc)
    
    # =========================================================================
    # PAGE 2: INDEX & ABSTRACT
    # =========================================================================
    add_chapter_title(doc, "INDEX & EXECUTIVE SUMMARY")
    add_section_title(doc, "Table of Contents")
    
    idx_tbl = doc.add_table(rows=10, cols=3)
    idx_tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(idx_tbl, color="CCCCCC", sz="4")
    
    headers = ["Sr. No.", "Chapter / Project Topic", "Page No."]
    for c_idx, h_text in enumerate(headers):
        cell = idx_tbl.cell(0, c_idx)
        set_cell_shading(cell, "EAECEF")
        set_cell_margins(cell, top=40, bottom=40, left=80, right=80)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER if c_idx != 1 else WD_ALIGN_PARAGRAPH.LEFT
        r = p.add_run(h_text)
        set_run_font(r, size_pt=9.0, bold=True)
        
    toc_data = [
        ("1", "Executive Summary & Abstract", "2"),
        ("2", "Chapter 1: Introduction of Project, Objectives and Scope", "3"),
        ("3", "Chapter 2: Introduction of Technologies Used & System Requirements", "4"),
        ("4", "Chapter 3: Dataset Specification, Characteristics & Preprocessing", "5"),
        ("5", "Chapter 4: Machine Learning Methodology & Training Pipeline", "6"),
        ("6", "Chapter 5: System Architecture & Full-Stack Implementation", "7"),
        ("7", "Chapter 6: Application User Interface & System Visualizations", "8"),
        ("8", "Chapter 7: Model Evaluation, Experimental Results and Testing", "9"),
        ("9", "Chapter 8: Conclusion, Limitations, Future Scope and References", "10")
    ]
    
    for r_idx, row_items in enumerate(toc_data, start=1):
        for c_idx, item in enumerate(row_items):
            cell = idx_tbl.cell(r_idx, c_idx)
            set_cell_margins(cell, top=30, bottom=30, left=80, right=80)
            if r_idx % 2 == 1:
                set_cell_shading(cell, "F9FAFB")
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if c_idx != 1 else WD_ALIGN_PARAGRAPH.LEFT
            r = p.add_run(item)
            set_run_font(r, size_pt=8.5, bold=False)
            
    add_para(doc, "Note: This index outlines the structural organization of this project report. Each chapter has been rigorously prepared in conformity with institutional documentation standards. Page numbers correspond directly to the physical printed layout.", 
             size_pt=8.0, italic=True, space_before=2, space_after=3)
    
    add_section_title(doc, "Executive Summary / Abstract")
    add_para(doc, "This project presents the design, empirical development, and production-grade deployment of the Wine Quality Prediction System — a decoupled, full-stack machine learning solution engineered to predict the organoleptic sensory quality scores of red wine from eleven laboratory physicochemical measurements. The underlying machine learning engine utilizes the benchmark UCI Wine Quality Dataset (Cortez et al., 2009), comprising 1,599 semicolon-delimited viticultural samples. The learning task is formulated as continuous numerical regression to preserve natural ordinal distances across sensory grades, avoiding the artificial discretization penalties characteristic of arbitrary classification bins.", 
             size_pt=9.0, space_after=2.5)
    add_para(doc, "Multiple predictive algorithms — including Multiple Linear Regression, Decision Tree Regressor, and Random Forest Regressor — were systematically trained, cross-validated via 5-fold cross-validation on 1,279 training records (80%), and evaluated on an isolated holdout test partition of 320 samples (20%). The optimized Random Forest ensemble, encapsulated within a standardized Scikit-Learn pipeline alongside robust z-score standardization, emerged as the superior model. It achieved a holdout Mean Absolute Error (MAE) of 0.4220, Root Mean Squared Error (RMSE) of 0.5483, and a Coefficient of Determination (R²) of 0.5399, effectively modeling complex non-linear physicochemical interactions.", 
             size_pt=9.0, space_after=2.5)
    add_para(doc, "To bridge academic modeling and practical utility, the trained pipeline is serialized and served via an asynchronous FastAPI RESTful backend equipped with strict Pydantic runtime schema contracts. The client interface is a reactive single-page dashboard built with Angular 21 and TypeScript, featuring a dark glassmorphic UI, dynamic feature sliders, real-time quality gauges, sensory presets, and instant visual analytics. The resulting system demonstrates an end-to-end software engineering workflow delivering accessible, automated, and mathematically verified enological decision support.", 
             size_pt=9.0, space_after=0)
    
    add_page_break(doc)
    
    # =========================================================================
    # PAGE 3: CHAPTER 1 - INTRODUCTION OF PROJECT
    # =========================================================================
    add_chapter_title(doc, "CHAPTER 1: INTRODUCTION OF PROJECT")
    
    add_section_title(doc, "1.1  Overview")
    add_para(doc, "Wine evaluation is traditionally governed by expert sensory panels who rate wines based on aroma, body, finish, and balance. While indispensable, human organoleptic testing is inherently subjective, expensive, resource-intensive, and prone to taster palate fatigue. In commercial enology, significant commercial value depends on maintaining consistent quality across batches. Modern analytical chemistry enables rapid laboratory profiling of physicochemical properties; however, translating these continuous chemical measurements into expected sensory scores requires sophisticated mathematical modeling. The Wine Quality Prediction System addresses this operational need by providing an automated, machine learning-driven web application that accurately estimates sensory quality from standard laboratory tests.")
    
    add_section_title(doc, "1.2  Purpose of the Project")
    add_para(doc, "The primary purpose of this project is to bridge the gap between academic machine learning pipelines and production web applications. While statistical modeling often remains confined to standalone Jupyter notebooks, real-world utility demands accessible, cross-platform deployment. This project delivers an integrated system allowing winemakers, quality assurance technicians, and students to enter wine attributes dynamically, receive instantaneous quality estimates with rounded score grades, and explore analytical insights through an intuitive, modern interface.")
    
    add_section_title(doc, "1.3  Problem Statement")
    add_para(doc, "In traditional viticultural operations, wine quality grading is either conducted post-fermentation via expensive human tasting panels or approximated using static spreadsheets that fail to capture non-linear biochemical interactions. For example, volatile acidity strongly degrades wine perception, yet its defect threshold varies based on alcohol volume and citric acid balance. Linear heuristics fail to capture these multidimensional dependencies. Furthermore, non-technical winery personnel lack accessible computational tools to evaluate experimental blends prior to bottling. A centralized, reliable, and real-time prediction system is therefore required.")
    
    add_section_title(doc, "1.4  Objectives of the System")
    add_bullet(doc, "1. Data Pipeline & Standardization: ", "Ingest the UCI Red Wine benchmark, enforce data integrity, and apply leakage-free feature scaling within cross-validation folds.")
    add_bullet(doc, "2. Multi-Model Benchmarking: ", "Train and compare Linear Regression, Decision Trees, and Random Forests using 5-fold cross-validation to select the optimal model.")
    add_bullet(doc, "3. High-Performance REST Gateway: ", "Develop a FastAPI gateway with Pydantic schema validation for robust, real-time HTTP inference.")
    add_bullet(doc, "4. Modern Presentation Layer: ", "Engineer a reactive Angular 21 interface featuring dynamic sliders, sensory presets, quality gauges, and diagnostic charts.")
    
    add_section_title(doc, "1.5  Scope, Target Users and System Limitations")
    add_para(doc, "The current scope encompasses red wine varieties evaluated across 11 standard physicochemical laboratory parameters. Target users include cellar technicians, enological researchers, and academic students. System limitations are acknowledged: predictions represent non-causal statistical associations rather than official certifications. Furthermore, human sensory scores carry natural subjective variance (~0.5 rating points), defining an empirical noise floor below which model error cannot be reduced without overfitting.")
    
    add_page_break(doc)
    
    # =========================================================================
    # PAGE 4: CHAPTER 2 - TECHNOLOGIES USED & REQUIREMENTS
    # =========================================================================
    add_chapter_title(doc, "CHAPTER 2: INTRODUCTION OF TECHNOLOGIES USED")
    
    add_section_title(doc, "2.1  Python (Programming Language) & Machine Learning Stack")
    add_para(doc, "Python 3.13 serves as the backbone for the machine learning architecture. Python was selected for its unmatched ecosystem of scientific computing libraries, syntax clarity, and widespread enterprise adoption. The core components of the machine learning pipeline include:")
    add_bullet(doc, "• Scikit-Learn (v1.9.0): ", "Used for data preprocessing (StandardScaler), predictive estimators (RandomForestRegressor, DecisionTreeRegressor, LinearRegression), cross-validation (KFold, GridSearchCV), and regression metrics calculation.")
    add_bullet(doc, "• Pandas (v2.3.3) & NumPy (v2.3.3): ", "Provides high-performance data structures (DataFrames), tabular manipulation, and optimized vectorized linear algebra operations for data splitting and preprocessing.")
    add_bullet(doc, "• Joblib (v1.5.3): ", "Provides high-throughput serialization of trained Scikit-Learn pipelines, enabling zero-overhead model persistence and instantaneous deployment in API environments.")
    
    add_section_title(doc, "2.2  FastAPI & Uvicorn (REST API Gateway)")
    add_para(doc, "FastAPI (v0.159) is a modern, high-concurrency web framework for building APIs with Python based on standard type hints and Starlette. It handles real-time HTTP prediction requests, enforces CORS policies, and automatically generates interactive OpenAPI/Swagger documentation. Pydantic V2 ensures strict request validation, verifying that incoming chemical properties reside within viable physical boundaries. Uvicorn acts as the high-speed ASGI web server hosting the application.")
    
    add_section_title(doc, "2.3  Angular 21 & TypeScript (Frontend Presentation Layer)")
    add_para(doc, "Angular 21 was selected to build an enterprise-grade single-page application (SPA). Leveraging TypeScript 5.9, standalone component architecture, and RxJS reactive streams, the frontend provides real-time state synchronization, smooth page transitions, and robust client-side validation without external monolithic UI libraries.")
    
    add_section_title(doc, "2.4  Hardware and Software Requirements Specification")
    
    req_tbl = doc.add_table(rows=8, cols=3)
    req_tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(req_tbl, color="CCCCCC", sz="4")
    
    req_headers = ["System Component", "Minimum Requirement", "Recommended Specification"]
    for c_idx, h_text in enumerate(req_headers):
        cell = req_tbl.cell(0, c_idx)
        set_cell_shading(cell, "EAECEF")
        set_cell_margins(cell, top=35, bottom=35, left=60, right=60)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER if c_idx != 0 else WD_ALIGN_PARAGRAPH.LEFT
        r = p.add_run(h_text)
        set_run_font(r, size_pt=8.5, bold=True)
        
    req_data = [
        ("Central Processor (CPU)", "Dual-Core x86_64 @ 2.0 GHz", "Intel Core i5 / AMD Ryzen 5 (4+ Cores)"),
        ("System Memory (RAM)", "4 GB DDR4", "8 GB – 16 GB High-Speed DDR4/DDR5"),
        ("Available Disk Space", "500 MB Free Storage", "2 GB Free SSD Space"),
        ("Operating System", "Windows 10 / Ubuntu 20.04 LTS", "Windows 11 64-bit / Linux x86_64"),
        ("Python Runtime", "Python 3.10.x", "Python 3.13.x 64-bit"),
        ("Node.js & Package Manager", "Node.js 18 LTS / npm 9", "Node.js 20 LTS / npm 10+"),
        ("Client Web Browser", "Google Chrome 90+ / MS Edge 90+", "Latest Google Chrome, Edge, or Firefox")
    ]
    
    for r_idx, row_items in enumerate(req_data, start=1):
        for c_idx, item in enumerate(row_items):
            cell = req_tbl.cell(r_idx, c_idx)
            set_cell_margins(cell, top=25, bottom=25, left=60, right=60)
            if r_idx % 2 == 1:
                set_cell_shading(cell, "F9FAFB")
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            r = p.add_run(item)
            set_run_font(r, size_pt=8.0, bold=False)
            
    add_para(doc, "The architecture requires no specialized GPU hardware or external relational database servers. In-memory data structures and compiled pipeline artifacts ensure sub-20ms inference response times on standard consumer workstations.", 
             size_pt=8.0, italic=True, space_before=2, space_after=0)
    
    add_page_break(doc)
    
    # =========================================================================
    # PAGE 5: CHAPTER 3 - DATASET SPECIFICATION & PREPROCESSING
    # =========================================================================
    add_chapter_title(doc, "CHAPTER 3: DATASET SPECIFICATION AND PREPROCESSING")
    
    add_section_title(doc, "3.1  Dataset Benchmark Specification")
    add_para(doc, "The system utilizes the authoritative UCI Wine Quality Dataset (Cortez et al., 2009). The dataset contains 1,599 red wine records sampled from the 'Vinho Verde' region of northern Portugal. Due to European formatting standards, the raw CSV file uses semicolon (';') delimiters, requiring specialized parsing logic: pd.read_csv(filepath, sep=';'). Each record contains 11 physicochemical laboratory measurements and a median sensory quality score.")
    
    add_section_title(doc, "3.2  Input Physicochemical Attributes and Target")
    
    feat_tbl = doc.add_table(rows=12, cols=4)
    feat_tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(feat_tbl, color="CCCCCC", sz="4")
    
    f_headers = ["#", "Feature Name", "Unit / Measurement", "Observed Range"]
    for c_idx, h_text in enumerate(f_headers):
        cell = feat_tbl.cell(0, c_idx)
        set_cell_shading(cell, "EAECEF")
        set_cell_margins(cell, top=30, bottom=30, left=50, right=50)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER if c_idx in [0, 3] else WD_ALIGN_PARAGRAPH.LEFT
        r = p.add_run(h_text)
        set_run_font(r, size_pt=8.0, bold=True)
        
    feat_data = [
        ("1", "fixed acidity", "Tartaric acid (g/dm³)", "4.6 – 15.9"),
        ("2", "volatile acidity", "Acetic acid (g/dm³) — vinegar defect", "0.12 – 1.58"),
        ("3", "citric acid", "Citric acid (g/dm³) — freshness", "0.00 – 1.00"),
        ("4", "residual sugar", "Post-fermentation sugar (g/dm³)", "0.9 – 15.5"),
        ("5", "chlorides", "Sodium chloride / salt (g/dm³)", "0.012 – 0.611"),
        ("6", "free sulfur dioxide", "Free molecular SO₂ gas (mg/dm³)", "1.0 – 72.0"),
        ("7", "total sulfur dioxide", "Free + bound SO₂ gas (mg/dm³)", "6.0 – 289.0"),
        ("8", "density", "Specific gravity (g/cm³)", "0.990 – 1.004"),
        ("9", "pH", "Wine acidity scale (0–14)", "2.74 – 4.01"),
        ("10", "sulphates", "Potassium sulphate (g/dm³)", "0.33 – 2.00"),
        ("11", "alcohol", "Alcohol volume (% vol)", "8.4 – 14.9")
    ]
    
    for r_idx, row_items in enumerate(feat_data, start=1):
        for c_idx, item in enumerate(row_items):
            cell = feat_tbl.cell(r_idx, c_idx)
            set_cell_margins(cell, top=20, bottom=20, left=50, right=50)
            if r_idx % 2 == 1:
                set_cell_shading(cell, "F9FAFB")
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if c_idx in [0, 3] else WD_ALIGN_PARAGRAPH.LEFT
            r = p.add_run(item)
            set_run_font(r, size_pt=7.5, bold=False)
            
    add_para(doc, "Regression Target: 'quality' is scored by certified enological experts on an integer scale from 0 (very bad) to 10 (very excellent). Observed scores span 3 to 8, with a median of 6.0 and mean of 5.64.", 
             size_pt=8.0, italic=True, space_before=1.5, space_after=2)
    
    add_section_title(doc, "3.3  Data Cleaning, Integrity Audit & Split Strategy")
    add_para(doc, "A comprehensive automated audit verified zero missing values, zero NaN cells, and zero non-numeric entries. While duplicate rows exist in the raw collection (arising from identical lab profiles across multiple commercial bottlings), retaining them reflects natural sampling frequency. An 80/20 train/test split was established with random_state=42, creating an isolated holdout test set of 320 samples (1,279 training samples) dedicated exclusively to final evaluation.")
    
    add_section_title(doc, "3.4  Feature Scaling Pipeline (Leakage-Free Transformation)")
    add_para(doc, "Because input variables span wildly disparate scales — from density (~1.0 g/cm³) to total sulfur dioxide (~289.0 mg/dm³) — features must be standardized. Standard z-score scaling: z = (x - μ) / σ is applied. Crucially, the StandardScaler is encapsulated inside a Scikit-Learn Pipeline, ensuring that mean (μ) and standard deviation (σ) are fitted solely on training partitions and applied to holdout test sets without statistical leakage.")
    
    add_page_break(doc)
    
    # =========================================================================
    # PAGE 6: CHAPTER 4 - MACHINE LEARNING METHODOLOGY
    # =========================================================================
    add_chapter_title(doc, "CHAPTER 4: MACHINE LEARNING METHODOLOGY")
    
    add_section_title(doc, "4.1  Regression Formulation vs. Classification Paradigm")
    add_para(doc, "While many textbook exercises treat wine quality as a classification problem by arbitrarily binning wines into 'good' (quality > 6) and 'bad' (quality <= 6), this project formulates quality prediction as continuous regression. Discretization discards valuable granular ordering information: misclassifying a quality 5 wine as quality 6 is a minor error of 1 point, whereas confusing quality 3 with quality 8 represents a catastrophic error of 5 points. Continuous regression respects these ordinal distances, outputs refined fractional scores (e.g., 5.82), and allows downstream rounding when integer grades are required.")
    
    add_section_title(doc, "4.2  Candidate Algorithms & Mathematical Foundations")
    add_bullet(doc, "• Multiple Linear Regression: ", "Establishes a baseline linear relationship: y = β₀ + Σ(βᵢ · xᵢ) + ε. While fast and interpretable, it assumes strict linearity and cannot capture threshold effects like sulfur dioxide saturation.")
    add_bullet(doc, "• Decision Tree Regressor: ", "Recursively partitions feature space using Mean Squared Error (MSE) minimization criteria. While capable of modeling non-linear thresholds, individual trees suffer from high variance and severe overfitting on small datasets.")
    add_bullet(doc, "• Random Forest Regressor (Selected Architecture): ", "An ensemble learning method that constructs B decorrelated decision trees using bootstrap aggregating (bagging) and random feature subspace sampling. The ensemble prediction averages individual tree outputs:")
    
    add_para(doc, "ŷ(x) = (1 / B) · Σ_{b=1}^{B} T_b(x)", align=WD_ALIGN_PARAGRAPH.CENTER, size_pt=10.0, bold=True, space_before=2, space_after=2)
    add_para(doc, "By aggregating hundreds of decorrelated decision trees, Random Forest dramatically reduces estimator variance without increasing bias, producing smooth, reliable generalization boundaries across complex chemical spaces.")
    
    add_section_title(doc, "4.3  Hyperparameter Tuning via GridSearchCV (5-Fold CV)")
    add_para(doc, "To maximize generalization performance without test leakage, systematic 5-fold cross-validation was executed strictly on the 1,279 training samples using GridSearchCV. The search space explored: n_estimators ∈ [50, 100, 200], max_depth ∈ [None, 10, 15], min_samples_split ∈ [2, 5], and min_samples_leaf ∈ [1, 2]. The optimal configuration (100 estimators, unconstrained depth, min_samples_split=2, min_samples_leaf=1) achieved the lowest mean cross-validation error, confirming robust ensemble convergence.")
    
    add_section_title(doc, "4.4  Model Serialization & Production Packaging")
    add_para(doc, "The finalized pipeline — encompassing both the fitted StandardScaler and the optimized RandomForestRegressor — was serialized using Joblib into models/wine_quality_model.joblib (3.4 MB). Comprehensive metadata detailing feature names, training dates, evaluation metrics, and hyperparameter dictionaries was exported to wine_quality_model_metadata.json, ensuring total operational reproducibility.")
    
    add_page_break(doc)
    
    # =========================================================================
    # PAGE 7: CHAPTER 5 - SYSTEM ARCHITECTURE AND IMPLEMENTATION
    # =========================================================================
    add_chapter_title(doc, "CHAPTER 5: SYSTEM ARCHITECTURE AND IMPLEMENTATION")
    
    add_section_title(doc, "5.1  Decoupled Multi-Tier System Architecture")
    add_para(doc, "The application adopts a decoupled client-server architecture cleanly segregating the graphical presentation layer from the analytical inference engine. The Angular 21 frontend communicates with the FastAPI backend over HTTP using structured JSON payloads. This decoupling ensures independent maintainability, zero UI coupling with ML logic, and easy scalability.")
    
    add_para(doc, "[ User Browser / Angular 21 UI ] ──(HTTP POST JSON)──> [ FastAPI REST Gateway ] ──(Pydantic Validation)──> [ Scikit-Learn Pipeline ] ──(Prediction)──> [ JSON Response ]", 
             align=WD_ALIGN_PARAGRAPH.CENTER, size_pt=8.0, bold=True, space_before=1.5, space_after=3)
             
    add_section_title(doc, "5.2  Backend REST API Endpoints & Request Validation")
    add_para(doc, "The FastAPI service exposes four core RESTful endpoints designed for production consumption:")
    add_bullet(doc, "• GET /health: ", "Verifies API operational status, uptime, and confirms active model artifact availability in memory.")
    add_bullet(doc, "• GET /model-info: ", "Inspects loaded model family, training sample count, and holdout regression performance metrics.")
    add_bullet(doc, "• GET /features: ", "Returns metadata, typical bounds, and chemical units for all 11 physicochemical features.")
    add_bullet(doc, "• POST /predict: ", "Ingests JSON payloads containing the 11 wine features, validates values via Pydantic, applies the StandardScaler, and returns continuous and rounded quality predictions with enological classification.")
    
    add_section_title(doc, "5.3  Angular Frontend Architecture & State Management")
    add_para(doc, "The frontend leverages Angular 21 standalone components, reactive forms, and RxJS observables to manage client-side state. Sensory presets allow users to populate typical chemical profiles with a single click, triggering instant prediction updates through reactive subscriptions.")
    
    add_section_title(doc, "5.4  Core Implementation Code Snippet")
    code_api_path = os.path.join(SCREENSHOTS_DIR, "code_api.png")
    if os.path.exists(code_api_path):
        add_framed_image(doc, code_api_path, width_inches=5.2, fig_label="Fig. 5.1", 
                         caption_text="main.py: Real-Time Inference Route with Pydantic Validation & Pipeline Execution")
                         
    add_page_break(doc)
    
    # =========================================================================
    # PAGE 8: CHAPTER 6 - APPLICATION UI AND ANALYTICS
    # =========================================================================
    add_chapter_title(doc, "CHAPTER 6: APPLICATION USER INTERFACE AND ANALYTICS")
    
    add_section_title(doc, "6.1  User Interface Design & Enological Workflow")
    add_para(doc, "The frontend was engineered with a modern dark glassmorphic design system tailored for laboratory and cellar environments. Key usability features include dynamic range sliders synchronized with numerical inputs, live field validation badges, sensory presets (e.g., Median Baseline, High Quality Blend) for instantaneous comparative testing, and an interactive quality gauge displaying predicted sensory scores.")
    
    add_section_title(doc, "6.2  Application Interface Screenshots")
    
    dash_path = os.path.join(SCREENSHOTS_DIR, "dashboard.png")
    pred_path = os.path.join(SCREENSHOTS_DIR, "prediction_result.png")
    
    if os.path.exists(dash_path):
        add_framed_image(doc, dash_path, width_inches=4.8, fig_label="Fig. 6.1", 
                         caption_text="Application Main Dashboard: Displaying Active Pipeline Status & Model Metrics")
    if os.path.exists(pred_path):
        add_framed_image(doc, pred_path, width_inches=4.8, fig_label="Fig. 6.2", 
                         caption_text="Sensory Prediction Interface: Feature Input Sliders, Quality Gauge & Output Card")
                         
    add_section_title(doc, "6.3  Interactive Features & Sensory Presets")
    add_para(doc, "The interface allows immediate toggling between benchmark wine profiles. Clicking 'Load Median Baseline' populates all 11 physicochemical inputs with verified dataset medians, immediately updating the real-time prediction gauge and displaying sensory quality tiers.")
    
    add_page_break(doc)
    
    # =========================================================================
    # PAGE 9: CHAPTER 7 - EXPERIMENTAL EVALUATION, RESULTS & TESTING
    # =========================================================================
    add_chapter_title(doc, "CHAPTER 7: EXPERIMENTAL EVALUATION, RESULTS AND TESTING")
    
    add_section_title(doc, "7.1  Regression Evaluation Metrics & Formulations")
    add_para(doc, "Model performance was comprehensively evaluated using four standard regression metrics:")
    add_bullet(doc, "• Mean Absolute Error (MAE): ", "Average absolute difference between predicted and true ratings on the 10-point scale: MAE = (1/n) · Σ |yᵢ - ŷᵢ|. Measured value: 0.4220.")
    add_bullet(doc, "• Root Mean Squared Error (RMSE): ", "Square root of MSE, penalizing larger deviations more severely: RMSE = √[(1/n) · Σ (yᵢ - ŷᵢ)²]. Measured value: 0.5483.")
    add_bullet(doc, "• Coefficient of Determination (R²): ", "Proportion of total variance in sensory scores explained by the 11 features: R² = 1 - [Σ(yᵢ - ŷᵢ)² / Σ(yᵢ - ȳ)²]. Measured value: 0.5399 (~54%).")
    
    add_section_title(doc, "7.2  Comparative Model Performance Results (Verified Evidence)")
    
    res_tbl = doc.add_table(rows=4, cols=7)
    res_tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(res_tbl, color="CCCCCC", sz="4")
    
    r_headers = ["Algorithm", "CV MAE", "CV RMSE", "CV R²", "Test MAE", "Test RMSE", "Test R²"]
    for c_idx, h_text in enumerate(r_headers):
        cell = res_tbl.cell(0, c_idx)
        set_cell_shading(cell, "EAECEF")
        set_cell_margins(cell, top=30, bottom=30, left=35, right=35)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER if c_idx != 0 else WD_ALIGN_PARAGRAPH.LEFT
        r = p.add_run(h_text)
        set_run_font(r, size_pt=7.5, bold=True)
        
    res_data = [
        ("Linear Regression", "0.5057", "0.6618", "0.3226", "0.5012", "0.6521", "0.3450"),
        ("Decision Tree Regressor", "0.4777", "0.8076", "-0.0065", "0.4688", "0.7786", "0.0712"),
        ("Random Forest (Selected)", "0.4319", "0.6008", "0.4420", "0.4220", "0.5483", "0.5399")
    ]
    
    for r_idx, row_items in enumerate(res_data, start=1):
        for c_idx, item in enumerate(row_items):
            cell = res_tbl.cell(r_idx, c_idx)
            set_cell_margins(cell, top=25, bottom=25, left=35, right=35)
            if r_idx % 2 == 1:
                set_cell_shading(cell, "F9FAFB")
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if c_idx != 0 else WD_ALIGN_PARAGRAPH.LEFT
            is_winner = (r_idx == 3)
            r = p.add_run(item)
            set_run_font(r, size_pt=7.5, bold=is_winner)
            
    add_para(doc, "Key Takeaway: The Random Forest Regressor substantially outperforms both Linear Regression and individual Decision Trees, achieving the lowest test error (MAE: 0.4220) and highest explained variance (R²: 0.5399).", 
             size_pt=8.0, italic=True, space_before=1.5, space_after=2)
             
    add_section_title(doc, "7.3  Diagnostic Plots & Feature Importance Ranking")
    
    avp_path = os.path.join(REPORTS_DIR, "actual_vs_predicted.png")
    if os.path.exists(avp_path):
        add_framed_image(doc, avp_path, width_inches=3.6, fig_label="Fig. 7.1", 
                         caption_text="Actual vs. Predicted Parity Plot: Scatter Distribution with 1:1 Identity Line")
                         
    add_para(doc, "Testing & Quality Assurance: An automated test suite comprising 39 unit and integration tests (pytest tests -v) comprehensively verified preprocessing boundaries, pipeline serialization, and FastAPI endpoint response contracts with 100% pass rates.", 
             size_pt=8.5, space_before=1, space_after=0)
             
    add_page_break(doc)
    
    # =========================================================================
    # PAGE 10: CHAPTER 8 - CONCLUSION, FUTURE SCOPE AND REFERENCES
    # =========================================================================
    add_chapter_title(doc, "CHAPTER 8: CONCLUSION, FUTURE SCOPE AND REFERENCES")
    
    add_section_title(doc, "8.1  Project Summary & Engineering Achievements")
    add_para(doc, "The Wine Quality Prediction System has been successfully conceptualized, developed, and benchmarked as an end-to-end full-stack machine learning solution. The project accomplished all primary engineering and academic objectives: ingesting the UCI Red Wine benchmark, preventing data leakage via an encapsulated Scikit-Learn pipeline, executing cross-validated hyperparameter optimization, and serving real-time predictions via an asynchronous FastAPI gateway to an Angular 21 reactive dashboard. The final Random Forest model achieved a test MAE of 0.4220 and an R² of 0.5399, validating the viability of automated sensory scoring from standard laboratory tests.")
    
    add_section_title(doc, "8.2  Key Learning Outcomes")
    add_para(doc, "Developing this system provided rigorous hands-on experience across multiple core disciplines: theoretical machine learning (ensemble bagging, bias-variance tradeoff), data leakage prevention, REST API architectural design, runtime schema validation with Pydantic, and modern reactive frontend development with Angular 21 and RxJS.")
    
    add_section_title(doc, "8.3  Identified Limitations & Challenges")
    add_bullet(doc, "• Class Imbalance at Score Tails: ", "True quality ratings in the UCI dataset are concentrated around 5 and 6 (82.5% of samples). Extreme scores (3 and 8) have scant representation (< 2% combined), restricting tail-end sensitivity.")
    add_bullet(doc, "• Human Sensory Noise Floor: ", "Human sensory ratings inherently carry inter-evaluator variance (~0.5 rating points). Consequently, residual errors near 0.42 reflect natural sensory variance rather than model deficiency.")
    add_bullet(doc, "• Omitted Enological Factors: ", "Crucial chemical drivers — such as tannin extractability, polyphenols, and grape varietal — are absent from the basic 11-feature benchmark.")
    
    add_section_title(doc, "8.4  Future Enhancements")
    add_bullet(doc, "• White Wine Multi-Model Extension: ", "Incorporate the companion 4,898-sample UCI White Wine dataset with dynamic varietal switching.")
    add_bullet(doc, "• Deep Tabular Architectures: ", "Explore modern deep learning tabular models (e.g., TabNet, FT-Transformer) with embeddings for non-linear feature interactions.")
    add_bullet(doc, "• Batch Processing & Winery Integration: ", "Implement bulk CSV upload capabilities to enable commercial wineries to screen entire bottling runs simultaneously.")
    
    add_section_title(doc, "8.5  References & Academic Bibliography")
    refs = [
        "Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009). Modeling wine preferences by data mining from physicochemical properties. Decision Support Systems, 47(4), 547-553.",
        "Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., et al. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12, 2825-2830.",
        "Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5-32.",
        "Ramirez, S. (2024). FastAPI: High performance, easy to learn, fast to code. Retrieved from https://fastapi.tiangolo.com/",
        "Google. (2024). Angular Documentation — Modern Web Development Framework. Retrieved from https://angular.dev/",
        "McKinney, W. (2010). Data Structures for Statistical Computing in Python. Proceedings of the 9th Python in Science Conference, 51-56."
    ]
    for r_text in refs:
        add_bullet(doc, "", r_text, space_after=1.2)
        
    print("Saving document to:", OUTPUT_DOCX)
    doc.save(OUTPUT_DOCX)
    print("Document saved successfully!")
    
    # Compute Page Count in MS Word
    try:
        print("Checking rendered page count via Microsoft Word COM...")
        word = win32com.client.Dispatch("Word.Application")
        word.Visible = False
        doc_com = word.Documents.Open(OUTPUT_DOCX)
        page_count = doc_com.ComputeStatistics(2) # 2 = wdStatisticPages
        print(f">>> EXACT RENDERED PAGE COUNT IN MS WORD: {page_count} <<<")
        doc_com.Close(False)
        word.Quit()
    except Exception as e:
        print("Word COM page count check:", e)

if __name__ == "__main__":
    main()
