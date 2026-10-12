"""Point-by-point response to the local-reviewer report received alongside the BMC Bioinformatics
scope-based rejection. References the actual revised manuscript's section/table/figure numbers."""
from docx import Document
from docx.shared import Pt, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn

doc = Document(); sec = doc.sections[0]
sec.left_margin = sec.right_margin = Cm(2.4); sec.top_margin = sec.bottom_margin = Cm(2.2)
st = doc.styles['Normal']; st.font.name = 'Times New Roman'; st.font.size = Pt(11)
st.element.rPr.rFonts.set(qn('w:eastAsia'), 'Times New Roman'); st.paragraph_format.space_after = Pt(8); st.paragraph_format.line_spacing = 1.25

t = doc.add_paragraph(); t.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = t.add_run('Response to Reviewer Comments'); r.bold = True; r.font.size = Pt(14)
p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.add_run('Manuscript: "Explanation Reliability Beyond Diagnostic Accuracy in Dermoscopic Skin-Lesion Classification"').italic = True
doc.add_paragraph()
doc.add_paragraph("We thank the reviewer for a careful and constructive report. The manuscript was declined by BMC Bioinformatics on scope grounds (the journal's own decision letter states this explicitly), but the reviewer's seven substantive points were all addressable and we have revised the manuscript accordingly. Below we respond to each point in turn, citing the specific section, table or figure in the revised manuscript where the change is made. Section/table/figure numbers refer to the revised manuscript.")


def comment(n, text):
    p = doc.add_paragraph(); r = p.add_run(f'Reviewer comment {n}. '); r.bold = True
    p.add_run(text); p.paragraph_format.space_before = Pt(14)


def response(text):
    p = doc.add_paragraph(); r = p.add_run('Response. '); r.bold = True
    p.add_run(text)


comment(1, "Recent literature is insufficiently covered, particularly recent studies on explainable skin-lesion classification, XAI evaluation, and external generalization.")
response("We have added three new Related Work subsections (2.4-2.6) covering, respectively: (i) quantitative saliency-map evaluation in medical imaging outside dermatology, including landmark radiology audits of saliency trustworthiness and localization benchmarking; (ii) external generalization and domain shift in dermoscopic and medical imaging AI, including the 2019 ISIC Grand Challenge (64 teams, 129 algorithms) and a systematic review of 86 externally validated radiologic deep-learning algorithms; and (iii) evaluation-protocol integrity in dermoscopic benchmarks, including a 2026 lesion-level leakage audit of HAM10000 and a data-quality audit of HAM10000/Fitzpatrick17k. Together with citations added to the Introduction and Discussion, this adds 20 new, individually verified references (2021-2026) spanning all three areas the reviewer identified, bringing the reference list from 62 to 81 entries.")

comment(2, "The research gap and contribution are not sufficiently differentiated from existing skin-lesion classification and explainability studies.")
response("Section 1.4 now includes a dedicated paragraph, “How this differs from recent dermoscopic XAI work,” that names and contrasts the present study against the closest recent work: dermatologist-agreement evaluation of saliency methods, morphometric analysis of saliency maps, harmonized multi-architecture evaluation frameworks, and composite faithfulness/robustness scoring. We state explicitly what none of these do that this study does — null-control the mask-overlap localization statistic against the geometry of the reference region — and draw an explicit parallel to, and gap relative to, analogous saliency-trustworthiness audits in radiology. Section 4.10 (new) and Section 5.8 (new) extend this differentiation with a direct, cited, tabulated comparison.")

comment(3, "The overall methodology/workflow is not presented clearly enough; a consolidated architecture/workflow diagram is needed to explain the complete experimental pipeline.")
response("Figure 1 has been redesigned from a schematic list of evaluation dimensions into a consolidated architecture/workflow diagram showing the complete pipeline end to end: the two source datasets, preprocessing, five-seed model training and calibration, Grad-CAM++ generation, the four parallel explanation-reliability audits, external evaluation and decomposition on PAD-UFES-20, multi-seed aggregation, and the resulting trust report. Section 3.1 now opens with an explicit pointer to this figure and a one-sentence map from the diagram's stages to the corresponding Methods subsections (3.2-3.13).")

comment(4, "The proposed methodological contribution is unclear. DenseNet121, Grad-CAM++, calibration, and other components are established methods, and the manuscript should more clearly identify what constitutes the actual methodological contribution.")
response("We have not changed the underlying claim — the contribution is evaluation methodology, not a new architecture or saliency method — but we have made it explicit and defensible with evidence rather than assertion. Section 1.4 states plainly that no individual component is claimed as novel, and specifies six concrete methodological contributions (points i-vi). Section 3.8 gives the mathematical derivation showing that the raw lesion-size effect on the localization score (LSAS) is expected by construction, not a property of the model, and Section 4.5 shows this empirically with a permutation test, a model-independent simulation, and multiple null/ceiling baselines; to our knowledge this null-controlled audit has not been reported previously for dermoscopic mask-overlap scores (Sections 1.4, 2.4). Section 5.9 (new) states explicitly which parts of the framework generalize beyond this specific model and dataset (the null-controlled localization audit, the faithfulness protocol, the external-decomposition method) and which do not.")

comment(5, "Comparison with recent state-of-the-art studies is insufficient, particularly in terms of classification performance and explainability evaluation.")
response("Section 4.10 (new) is a dedicated Results subsection built around Table 16 (new), which compares the present study against 10 recent studies (2022-2026) on architecture, dataset, split protocol, accuracy/balanced accuracy, macro-F1, AUROC, multi-seed status, external validation and the type of explainability evaluation performed. The comparison is deliberately not presented as like-for-like: split protocol (lesion-level vs. image-level vs. unstated) is tabulated explicitly wherever it could be determined from each cited study, and the accompanying text draws out three evidence-based patterns — the association between higher reported accuracy and unstated (or leakage-prone) split protocols, the greater sensitivity of macro-F1 than accuracy to protocol and to class imbalance, and the near-universal absence of null-controlled localization evaluation in the compared explainability literature.")

comment(6, "The experimental results require stronger critical discussion, especially the relatively modest classification performance and substantial performance degradation on the external PAD-UFES-20 dataset.")
response("Two new Discussion subsections address this directly. Section 5.8, “Comparative interpretation of measured performance and external generalization,” argues from cited evidence (not assertion) that the reported accuracy and macro-F1 are consistent with, and in one direct independent replication almost identical to, verified lesion-level multi-seed HAM10000 baselines, and that the higher accuracies frequently reported in recent literature are associated with split protocols that are either unstated or, where audited independently, leakage-prone. The same subsection interprets the PAD-UFES-20 drop against the largest controlled multi-team dermoscopy benchmark to date (balanced accuracy falling from 82.0% to 58.8% across datasets) and against a systematic review finding that 81% of 86 externally validated radiologic algorithms show some performance decrease, arguing that the present drop is consistent with, rather than exceptional relative to, this literature. We also identify and discuss a single recent study reporting near-ceiling cross-dataset retention on the same dataset pair, and explain why it is treated as an unreplicated outlier claim rather than a benchmark.")

comment(7, "The practical significance and limitations of the proposed evaluation framework are not sufficiently established, particularly regarding its generalizability and clinical relevance.")
response("Section 5.9 (new), “Practical significance, generalizability and limitations of the evaluation framework,” states explicitly which parts of the framework are architecture-, saliency-method- and dataset-independent (the geometric null-control mathematics, the perturbation-faithfulness protocol, the external-decomposition method) and draws a concrete, cited parallel to an analogous but previously unexplained finding in chest-radiograph saliency evaluation. It also states the framework's boundaries without qualification: it requires a reference segmentation, is computationally more expensive than reporting a raw localization score, has not been tested on attention-based explanations, and — most importantly for clinical relevance — audits explanation reliability as a measurable statistical property and does not by itself establish clinical usefulness, for which a dermatologist reader study would be required. This complements, rather than duplicates, the existing Limitations section (Section 6), which lists study-specific rather than framework-level limitations.")

doc.add_paragraph().paragraph_format.space_before = Pt(16)
p = doc.add_paragraph("We believe these revisions address each point raised, and thank the reviewer again for comments that have made the manuscript substantially stronger.")
doc.save('out/Response_to_Reviewers.docx')
print('response letter built')
