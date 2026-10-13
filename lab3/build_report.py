"""Сборка отчёта lab3: Word с формулами OMML + экспорт в PDF через Pages."""
from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor
from lxml import etree

LAB3 = Path(__file__).resolve().parent
DOCX = LAB3 / "report.docx"
PDF = LAB3 / "report.pdf"

M_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"


def m(tag: str) -> str:
    return f"{{{M_NS}}}{tag}"


def omml_plain(text: str):
    r = etree.Element(m("r"))
    rPr = etree.SubElement(r, m("rPr"))
    sty = etree.SubElement(rPr, m("sty"))
    sty.set(qn("m:val"), "p")
    t = etree.SubElement(r, m("t"))
    t.text = text
    t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    return r


def append_parts(parent, parts):
    for p in parts:
        if isinstance(p, str):
            parent.append(omml_plain(p))
        else:
            parent.append(p)


def omml_sup(base, exp):
    sSup = etree.Element(m("sSup"))
    e = etree.SubElement(sSup, m("e"))
    append_parts(e, [base] if not isinstance(base, list) else base)
    sup = etree.SubElement(sSup, m("sup"))
    append_parts(sup, [exp] if not isinstance(exp, list) else exp)
    return sSup


def omml_frac(num, den):
    f = etree.Element(m("f"))
    num_el = etree.SubElement(f, m("num"))
    den_el = etree.SubElement(f, m("den"))
    append_parts(num_el, num if isinstance(num, list) else [num])
    append_parts(den_el, den if isinstance(den, list) else [den])
    return f


def omml_sqrt(inner):
    rad = etree.Element(m("rad"))
    rad_pr = etree.SubElement(rad, m("radPr"))
    deg_hide = etree.SubElement(rad_pr, m("degHide"))
    deg_hide.set(qn("m:val"), "1")
    etree.SubElement(rad, m("deg"))
    e = etree.SubElement(rad, m("e"))
    append_parts(e, inner if isinstance(inner, list) else [inner])
    return rad


def make_omath(*parts):
    o_math = etree.Element(m("oMath"))
    append_parts(o_math, parts)
    return o_math


def insert_equation(paragraph, *parts):
    o_math_para = etree.Element(m("oMathPara"))
    o_math_para.append(make_omath(*parts))
    paragraph._p.append(o_math_para)


def set_run_font(run, size=11, bold=False):
    run.font.name = "Times New Roman"
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    run.font.size = Pt(size)
    run.bold = bold


def add_text(doc, text, size=11, bold=False, center=False, space_after=6):
    p = doc.add_paragraph()
    if center:
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.space_before = Pt(2)
    run = p.add_run(text)
    set_run_font(run, size=size, bold=bold)
    return p


def add_heading_custom(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(12)
    p.paragraph_format.space_after = Pt(6)
    run = p.add_run(text)
    set_run_font(run, size=13, bold=True)
    run.font.color.rgb = RGBColor(0x0B, 0x3D, 0x2E)
    return p


def add_eq(doc, *parts):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(4)
    insert_equation(p, *parts)
    return p


def build() -> Path:
    doc = Document()
    for section in doc.sections:
        section.top_margin = Cm(2)
        section.bottom_margin = Cm(2)
        section.left_margin = Cm(2.5)
        section.right_margin = Cm(2.5)

    for _ in range(3):
        doc.add_paragraph()

    add_text(doc, "Практическое задание", size=12, center=True, space_after=4)
    add_text(doc, "Метод множителей Лагранжа", size=20, bold=True, center=True, space_after=8)
    add_text(doc, "Лабораторная работа №3, вариант 7", size=12, center=True, space_after=16)
    add_text(doc, "Соколов Никита Константинович", size=12, center=True, space_after=4)
    add_text(doc, "Группа К3339, поток 1.2", size=12, center=True, space_after=16)

    table = doc.add_table(rows=6, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    lines = [
        "Состав отчёта",
        "1. Постановка задачи",
        "2. Анализ применимости",
        "3. Ручное решение",
        "4. Кандидаты и глобальность",
        "5. Вывод",
    ]
    for i, line in enumerate(lines):
        cell = table.rows[i].cells[0]
        cell.text = ""
        run = cell.paragraphs[0].add_run(line)
        set_run_font(run, size=10, bold=(i == 0))

    tbl = table._tbl
    tbl_pr = tbl.tblPr if tbl.tblPr is not None else OxmlElement("w:tblPr")
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right"):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "single")
        el.set(qn("w:sz"), "12")
        el.set(qn("w:color"), "0B3D2E")
        borders.append(el)
    for edge in ("insideH", "insideV"):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "nil")
        borders.append(el)
    tbl_pr.append(borders)

    doc.add_paragraph()
    add_text(doc, "Ответ:", size=11, center=True, space_after=2)
    add_eq(
        doc,
        "min f = -3",
        omml_sqrt("3"),
        "  в точке  (-",
        omml_sqrt("3"),
        ", -",
        omml_sqrt("3"),
        ", -",
        omml_sqrt("3"),
        ")",
    )
    add_eq(
        doc,
        "max f = 3",
        omml_sqrt("3"),
        "  в точке  (",
        omml_sqrt("3"),
        ", ",
        omml_sqrt("3"),
        ", ",
        omml_sqrt("3"),
        ")",
    )

    doc.add_page_break()

    add_heading_custom(doc, "1. Постановка задачи")
    add_text(doc, "Найти глобальные минимум и максимум функции")
    add_eq(doc, "f(x, y, z) = x + y + z")
    add_text(doc, "при ограничении")
    add_eq(doc, omml_sup("x", "2"), " + ", omml_sup("y", "2"), " + ", omml_sup("z", "2"), " = 9")
    add_text(doc, "или в виде h = 0:")
    add_eq(
        doc,
        "h(x, y, z) = ",
        omml_sup("x", "2"),
        " + ",
        omml_sup("y", "2"),
        " + ",
        omml_sup("z", "2"),
        " - 9 = 0",
    )
    add_text(doc, "Переменные x, y, z действительные.")

    add_heading_custom(doc, "2. Анализ применимости")
    add_text(doc, "1. Функции f и h — многочлены, значит гладкие.")
    add_text(doc, "2. Допустимое множество — сфера радиуса 3. Оно непусто, замкнуто и ограничено.")
    add_text(doc, "3. Градиент ограничения:")
    add_eq(doc, "∇h = (2x, 2y, 2z)")
    add_text(
        doc,
        "На сфере ∇h = 0 только в (0, 0, 0), но эта точка не лежит на сфере. "
        "Значит на допустимом множестве ранг равен 1, все точки регулярны.",
    )
    add_text(
        doc,
        "4. Метод множителей Лагранжа применим: он даёт необходимые условия экстремума. "
        "Глобальность обосновывается сравнением значений f во всех найденных кандидатах.",
    )

    add_heading_custom(doc, "3. Ручное решение")
    add_text(doc, "Функция Лагранжа:")
    add_eq(
        doc,
        "L(x, y, z, λ) = x + y + z + λ(",
        omml_sup("x", "2"),
        " + ",
        omml_sup("y", "2"),
        " + ",
        omml_sup("z", "2"),
        " - 9)",
    )
    add_text(doc, "Система стационарности:")
    add_eq(doc, "∂L/∂x = 1 + 2λx = 0")
    add_eq(doc, "∂L/∂y = 1 + 2λy = 0")
    add_eq(doc, "∂L/∂z = 1 + 2λz = 0")
    add_eq(doc, omml_sup("x", "2"), " + ", omml_sup("y", "2"), " + ", omml_sup("z", "2"), " - 9 = 0")
    add_text(doc, "Из первых трёх уравнений при λ ≠ 0:")
    add_eq(doc, "x = y = z = ", omml_frac("-1", "2λ"))
    add_text(doc, "Подстановка в ограничение:")
    add_eq(doc, "3 · ", omml_frac("1", ["4", omml_sup("λ", "2")]), " = 9")
    add_eq(doc, omml_frac("1", ["4", omml_sup("λ", "2")]), " = 3")
    add_eq(doc, omml_sup("λ", "2"), " = ", omml_frac("1", "12"))
    add_eq(
        doc,
        "λ = ± ",
        omml_frac("1", ["2", omml_sqrt("3")]),
        " = ± ",
        omml_frac(omml_sqrt("3"), "6"),
    )

    add_heading_custom(doc, "4. Кандидаты и глобальность")
    add_text(doc, "При")
    add_eq(doc, "λ = ", omml_frac(omml_sqrt("3"), "6"))
    add_eq(doc, "x = y = z = -", omml_sqrt("3"))
    add_eq(doc, "f = -3", omml_sqrt("3"), " ≈ -5.196152")

    add_text(doc, "При")
    add_eq(doc, "λ = -", omml_frac(omml_sqrt("3"), "6"))
    add_eq(doc, "x = y = z = ", omml_sqrt("3"))
    add_eq(doc, "f = 3", omml_sqrt("3"), " ≈ 5.196152")

    add_text(doc, "Таблица кандидатов:", space_after=4)
    ct = doc.add_table(rows=3, cols=6)
    ct.style = "Table Grid"
    headers = ["№", "λ", "Точка (x, y, z)", "f", "Невязка h", "Классификация"]
    rows = [
        ["1", "√3/6", "(−√3, −√3, −√3)", "−3√3", "0", "глобальный минимум"],
        ["2", "−√3/6", "(√3, √3, √3)", "3√3", "0", "глобальный максимум"],
    ]
    for j, h in enumerate(headers):
        cell = ct.rows[0].cells[j]
        cell.text = ""
        run = cell.paragraphs[0].add_run(h)
        set_run_font(run, size=9, bold=True)
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            cell = ct.rows[i + 1].cells[j]
            cell.text = ""
            run = cell.paragraphs[0].add_run(val)
            set_run_font(run, size=9)

    doc.add_paragraph()
    add_text(doc, "Проверка ограничения для обеих точек:")
    add_eq(doc, "3 · (", omml_sup(omml_sqrt("3"), "2"), ") = 3 · 3 = 9")
    add_text(doc, "Других действительных решений системы нет: из уравнений следует x = y = z.")
    add_text(
        doc,
        "Глобальность: допустимое множество компактно, f непрерывна, поэтому глобальные "
        "минимум и максимум существуют. После сравнения значений f во всех кандидатах:",
    )
    add_eq(
        doc,
        "глобальный минимум: (-",
        omml_sqrt("3"),
        ", -",
        omml_sqrt("3"),
        ", -",
        omml_sqrt("3"),
        "),  f = -3",
        omml_sqrt("3"),
    )
    add_eq(
        doc,
        "глобальный максимум: (",
        omml_sqrt("3"),
        ", ",
        omml_sqrt("3"),
        ", ",
        omml_sqrt("3"),
        "),  f = 3",
        omml_sqrt("3"),
    )

    add_heading_custom(doc, "5. Вывод")
    add_text(doc, "1. Метод множителей Лагранжа применим.")
    add_text(doc, "2. Найдены два кандидата.")
    add_text(doc, "3. Глобальный минимум: f = −3√3 в точке (−√3, −√3, −√3).")
    add_text(doc, "4. Глобальный максимум: f = 3√3 в точке (√3, √3, √3).")
    add_text(
        doc,
        "5. Код программы: lab3/lagrange_v7.py "
        "(https://github.com/ProgrammistNik/Optimization_Methods)",
    )

    doc.save(str(DOCX))
    print("saved", DOCX)
    return DOCX


if __name__ == "__main__":
    build()
