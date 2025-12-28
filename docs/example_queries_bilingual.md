# CareMind · Example Queries (Bilingual)

Use these copy/paste queries in the app UI. Each row includes an English prompt and a Chinese equivalent.

Optional: specify a medicine name to make the answer more specific.
- You may omit the whole “Medicine …” line if you don’t have a specific drug.
- Please use the exact medicine name as it appears in your drug sources: `data/drugs.xlsx` (Excel source) or `db/drugs.sqlite` (SQLite DB).
- In the examples below, the medicine name is filled in (no blanks). Replace it with a medicine from your own drug source when needed.
- Place the medicine name after the question.
- English format: `<question>` then `Medicine (optional): <name>`
- 中文格式：先写`问题`，再写`药品名称（可选）：<药名>`

## Note on Medicine Names

The medicine names in this table are **placeholders for demo/testing** (to trigger the optional drug lookup) and are **not prescriptions or “recommended drugs”**.

Some entries are intentionally generic and may be **clinically mismatched** with the question. Replace the medicine with what you actually want to ask about, or leave it blank.

Examples of potential mismatches in this table:
- **Gout flare**: Aspirin is generally **not** a first-line treatment and can worsen uric acid; consider using a more relevant example (e.g., colchicine/NSAID) or leave blank.
- **GDM insulin initiation**: Metformin is not insulin; consider using an insulin product name (or leave blank) if the goal is insulin initiation.
- **DKD screening frequency**: Metformin is not directly relevant to albuminuria/eGFR screening; consider leaving blank or using a kidney-protective agent if that’s your focus.

| # | English | 中文 |
|---:|---|---|
| 1 | Can β-blockers be used in patients with hypertension who also have bronchial asthma?<br>Medicine (optional): Metoprolol | 合并支气管哮喘的高血压患者是否可用β受体阻滞剂？<br>药品名称（可选）：美托洛尔 |
| 2 | In chronic kidney disease (CKD), how should ACEI/ARB therapy be monitored (labs and follow-up interval)?<br>Medicine (optional): Enalapril | 慢性肾病（CKD）患者使用 ACEI/ARB 时应如何监测（检查项目与复查频率）？<br>药品名称（可选）：依那普利 |
| 3 | For an elderly patient with type 2 diabetes and coronary artery disease, what is the recommended blood pressure target and first-line antihypertensive choice?<br>Medicine (optional): Amlodipine | 老年合并 2 型糖尿病与冠心病患者，推荐的降压目标与首选降压方案是什么？<br>药品名称（可选）：氨氯地平 |
| 4 | In pregnancy (gestational diabetes), when should insulin be started, and what are typical starting strategies?<br>Medicine (optional): Metformin | 妊娠期糖尿病（GDM）何时需要起始胰岛素？常见的起始策略是什么？<br>药品名称（可选）：二甲双胍 |
| 5 | For atrial fibrillation with high stroke risk, how do guidelines suggest choosing anticoagulation and assessing bleeding risk?<br>Medicine (optional): Rivaroxaban | 房颤且卒中风险较高时，指南如何建议选择抗凝治疗并评估出血风险？<br>药品名称（可选）：利伐沙班 |
| 6 | In heart failure with reduced ejection fraction (HFrEF), what core medication classes are recommended and how are they titrated?<br>Medicine (optional): Spironolactone | 射血分数降低的心衰（HFrEF）推荐的核心药物类别有哪些？如何逐步加量？<br>药品名称（可选）：螺内酯 |
| 7 | For community-acquired pneumonia in adults, what empiric antibiotic options are recommended based on severity and comorbidities?<br>Medicine (optional): Amoxicillin/Clavulanate | 成人社区获得性肺炎（CAP）按严重程度与合并症分层时，经验性抗菌治疗有哪些推荐方案？<br>药品名称（可选）：阿莫西林/克拉维酸 |
| 8 | In patients with diabetes, what is recommended screening frequency for diabetic kidney disease (albuminuria/eGFR) and how should abnormal results be managed?<br>Medicine (optional): Metformin | 糖尿病患者应多长时间筛查糖尿病肾病（尿白蛋白/肌酐比、eGFR）？异常结果如何处理？<br>药品名称（可选）：二甲双胍 |
| 9 | For acute gout flare, what first-line treatments are recommended and what key contraindications should be considered?<br>Medicine (optional): Aspirin | 急性痛风发作时，一线治疗推荐是什么？需要关注哪些重要禁忌证？<br>药品名称（可选）：阿司匹林 |
| 10 | For suspected upper gastrointestinal bleeding, what initial stabilization steps and risk stratification approaches are recommended?<br>Medicine (optional): Omeprazole | 怀疑上消化道出血时，指南推荐的初始处理（复苏/稳定）与风险分层要点是什么？<br>药品名称（可选）：奥美拉唑 |
| 11 | For COPD exacerbation, what are recommended bronchodilator, steroid, and antibiotic indications?<br>Medicine (optional): Formoterol | 慢阻肺急性加重时，支气管扩张剂、糖皮质激素及抗生素的适应证是什么？<br>药品名称（可选）：福莫特罗 |
| 12 | In hyperlipidemia, how do guidelines recommend deciding statin intensity for primary prevention based on risk?<br>Medicine (optional): Atorvastatin | 高脂血症的一级预防中，指南如何基于风险分层选择他汀治疗强度？<br>药品名称（可选）：阿托伐他汀 |
