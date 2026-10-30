# lab4 — Пошаговое управление инвестиционным портфелем

Динамическое программирование, соотношение Беллмана.

## Запуск

```bash
cd lab4
python3 portfolio_dp.py
python3 outcome_tree.py
python3 build_report.py   # пересборка report.pdf
```

## Файлы

| Файл | Назначение |
|------|------------|
| `portfolio_dp.py` | программа (Беллман, тесты T1–T9, §8.3) |
| `build_report.py` | сборка отчёта (reportlab) |
| `report.pdf` | отчёт |
| `class_diagram.png` | диаграмма классов |
| `outcome_tree.py` | доп.: дерево исходов |
| `outcome_tree.txt` | результат дерева |

## Ответ

Критерий ожидания: **I\* ≈ 134.24 д.е.**, уникальное u₁*=(23,0,0) среди |A(s₀)|=1502.
