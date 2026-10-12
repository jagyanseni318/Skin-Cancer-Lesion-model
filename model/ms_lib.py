"""Manuscript builder library: citation manager (numbers assigned in order of first citation), verified reference dictionary,
markdown + docx writers. All reference entries were verified against Crossref / arXiv metadata during the revision."""
import re
from pathlib import Path
from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

REFS = {
 'esteva2017': "Esteva A, Kuprel B, Novoa RA, et al. Dermatologist-level classification of skin cancer with deep neural networks. Nature. 2017;542:115–118. doi:10.1038/nature21056.",
 'liu2020': "Liu Y, Jain A, Eng C, et al. A deep learning system for differential diagnosis of skin diseases. Nat Med. 2020;26:900–908. doi:10.1038/s41591-020-0842-3.",
 'tschandl2020': "Tschandl P, Rinner C, Apalla Z, et al. Human–computer collaboration for skin cancer recognition. Nat Med. 2020;26:1229–1234. doi:10.1038/s41591-020-0942-0.",
 'haenssle2018': "Haenssle HA, Fink C, Schneiderbauer R, et al. Man against machine: diagnostic performance of a deep learning convolutional neural network for dermoscopic melanoma recognition in comparison to 58 dermatologists. Ann Oncol. 2018;29:1836–1842. doi:10.1093/annonc/mdy166.",
 'tschandl2019': "Tschandl P, Codella N, Akay BN, et al. Comparison of the accuracy of human readers versus machine-learning algorithms for pigmented skin lesion classification: an open, web-based, international, diagnostic study. Lancet Oncol. 2019;20:938–947. doi:10.1016/S1470-2045(19)30333-X.",
 'ham10000': "Tschandl P, Rosendahl C, Kittler H. The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. Sci Data. 2018;5:180161. doi:10.1038/sdata.2018.161.",
 'isic2018': "Codella N, Rotemberg V, Tschandl P, et al. Skin lesion analysis toward melanoma detection 2018: a challenge hosted by the International Skin Imaging Collaboration (ISIC). arXiv:1902.03368. 2019.",
 'pad': "Pacheco AGC, Lima GR, Salomão AS, et al. PAD-UFES-20: a skin lesion dataset composed of patient data and clinical images collected from smartphones. Data Brief. 2020;32:106221. doi:10.1016/j.dib.2020.106221.",
 'vandervelden2022': "van der Velden BHM, Kuijf HJ, Gilhuijs KGA, Viergever MA. Explainable artificial intelligence (XAI) in deep learning-based medical image analysis. Med Image Anal. 2022;79:102470. doi:10.1016/j.media.2022.102470.",
 'hauser2022': "Hauser K, Kurz A, Haggenmüller S, et al. Explainable artificial intelligence in skin cancer recognition: a systematic review. Eur J Cancer. 2022;167:54–69. doi:10.1016/j.ejca.2022.02.025.",
 'chanda2024': "Chanda T, Hauser K, Hobelsberger S, et al. Dermatologist-like explainable AI enhances trust and confidence in diagnosing melanoma. Nat Commun. 2024;15. doi:10.1038/s41467-023-43095-4.",
 'gradcam': "Selvaraju RR, Cogswell M, Das A, Vedantam R, Parikh D, Batra D. Grad-CAM: visual explanations from deep networks via gradient-based localization. Proc IEEE ICCV. 2017:618–626. doi:10.1109/ICCV.2017.74.",
 'gradcampp': "Chattopadhay A, Sarkar A, Howlader P, Balasubramanian VN. Grad-CAM++: generalized gradient-based visual explanations for deep convolutional networks. Proc IEEE WACV. 2018:839–847. doi:10.1109/WACV.2018.00097.",
 'rise': "Petsiuk V, Das A, Saenko K. RISE: randomized input sampling for explanation of black-box models. Proc British Machine Vision Conference (BMVC). 2018. arXiv:1806.07421.",
 'samek2017': "Samek W, Binder A, Montavon G, Lapuschkin S, Müller KR. Evaluating the visualization of what a deep neural network has learned. IEEE Trans Neural Netw Learn Syst. 2017;28:2660–2673. doi:10.1109/TNNLS.2016.2599820.",
 'arras2022': "Arras L, Osman A, Samek W. CLEVR-XAI: a benchmark dataset for the ground truth evaluation of neural network explanations. Inf Fusion. 2022;81:14–40. doi:10.1016/j.inffus.2021.11.008.",
 'adebayo2018': "Adebayo J, Gilmer J, Muelly M, Goodfellow I, Hardt M, Kim B. Sanity checks for saliency maps. Adv Neural Inf Process Syst. 2018;31.",
 'hooker2019': "Hooker S, Erhan D, Kindermans PJ, Kim B. A benchmark for interpretability methods in deep neural networks. Adv Neural Inf Process Syst. 2019;32.",
 'rong2022': "Rong Y, Leemann T, Borisov V, Kasneci G, Kasneci E. A consistent and efficient evaluation strategy for attribution methods. Proc ICML. 2022;162. arXiv:2202.00449.",
 'yeh2019': "Yeh CK, Hsieh CY, Suggala AS, Inouye DI, Ravikumar P. On the (in)fidelity and sensitivity for explanations. Adv Neural Inf Process Syst. 2019;32. arXiv:1901.09392.",
 'ghassemi2021': "Ghassemi M, Oakden-Rayner L, Beam AL. The false hope of current approaches to explainable artificial intelligence in health care. Lancet Digit Health. 2021;3:e745–e750. doi:10.1016/S2589-7500(21)00208-9.",
 'rudin2019': "Rudin C. Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead. Nat Mach Intell. 2019;1:206–215. doi:10.1038/s42256-019-0048-x.",
 'degrave2021': "DeGrave AJ, Janizek JD, Lee SI. AI for radiographic COVID-19 detection selects shortcuts over signal. Nat Mach Intell. 2021;3:610–619. doi:10.1038/s42256-021-00338-7.",
 'bissoto2019': "Bissoto A, Fornaciali M, Valle E, Avila S. (De)constructing bias on skin lesion datasets. Proc IEEE/CVF CVPR Workshops. 2019:2766–2774. doi:10.1109/CVPRW.2019.00335.",
 'winkler2019': "Winkler JK, Fink C, Toberer F, et al. Association between surgical skin markings in dermoscopic images and diagnostic performance of a deep learning convolutional neural network for melanoma recognition. JAMA Dermatol. 2019;155:1135. doi:10.1001/jamadermatol.2019.1735.",
 'guo2017': "Guo C, Pleiss G, Sun Y, Weinberger KQ. On calibration of modern neural networks. Proc ICML. 2017;70:1321–1330.",
 'naeini2015': "Naeini MP, Cooper GF, Hauskrecht M. Obtaining well calibrated probabilities using Bayesian binning. Proc AAAI. 2015;29. doi:10.1609/aaai.v29i1.9602.",
 'brier1950': "Brier GW. Verification of forecasts expressed in terms of probability. Mon Weather Rev. 1950;78:1–3. doi:10.1175/1520-0493(1950)078<0001:VOFEIT>2.0.CO;2.",
 'gal2016': "Gal Y, Ghahramani Z. Dropout as a Bayesian approximation: representing model uncertainty in deep learning. Proc ICML. 2016;48:1050–1059.",
 'ovadia2019': "Ovadia Y, Fertig E, Ren J, et al. Can you trust your model's uncertainty? Evaluating predictive uncertainty under dataset shift. Adv Neural Inf Process Syst. 2019;32:13991–14002.",
 'huang2017': "Huang G, Liu Z, van der Maaten L, Weinberger KQ. Densely connected convolutional networks. Proc IEEE CVPR. 2017:2261–2269. doi:10.1109/CVPR.2017.243.",
 'lin2020': "Lin TY, Goyal P, Girshick R, He K, Dollár P. Focal loss for dense object detection. IEEE Trans Pattern Anal Mach Intell. 2020;42:318–327. doi:10.1109/TPAMI.2018.2858826.",
 'smith2019': "Smith LN, Topin N. Super-convergence: very fast training of neural networks using large learning rates. Proc SPIE 11006, Artificial Intelligence and Machine Learning for Multi-Domain Operations Applications. 2019;1100612. doi:10.1117/12.2520589.",
 'loshchilov2019': "Loshchilov I, Hutter F. Decoupled weight decay regularization. Proc ICLR. 2019. arXiv:1711.05101.",
 'lee1997': "Lee T, Ng V, Gallagher R, Coldman A, McLean D. DullRazor: a software approach to hair removal from images. Comput Biol Med. 1997;27:533–543. doi:10.1016/S0010-4825(97)00020-6.",
 'chardon1991': "Chardon A, Cretois I, Hourseau C. Skin colour typology and suntanning pathways. Int J Cosmet Sci. 1991;13:191–208. doi:10.1111/j.1467-2494.1991.tb00561.x.",
 'delbino2013': "Del Bino S, Bernerd F. Variations in skin colour and the biological consequences of ultraviolet radiation exposure. Br J Dermatol. 2013;169(Suppl 3):33–40. doi:10.1111/bjd.12529.",
 'fitzpatrick1988': "Fitzpatrick TB. The validity and practicality of sun-reactive skin types I through VI. Arch Dermatol. 1988;124:869–871. doi:10.1001/archderm.1988.01670060015008.",
 'daneshjou2022': "Daneshjou R, Vodrahalli K, Novoa RA, et al. Disparities in dermatology AI performance on a diverse, curated clinical image set. Sci Adv. 2022;8:eabq6147. doi:10.1126/sciadv.abq6147.",
 'groh2021': "Groh M, Harris C, Soenksen L, et al. Evaluating deep neural networks trained on clinical images in dermatology with the Fitzpatrick 17k dataset. Proc IEEE/CVF CVPR Workshops. 2021:1820–1828. doi:10.1109/CVPRW53098.2021.00201.",
 'groh2024': "Groh M, Badri O, Daneshjou R, et al. Deep learning-aided decision support for diagnosis of skin disease across skin tones. Nat Med. 2024;30:573–583. doi:10.1038/s41591-023-02728-3.",
 'kinyanjui2020': "Kinyanjui NM, Odonga T, Cintas C, et al. Fairness of classifiers across skin tones in dermatology. In: Medical Image Computing and Computer Assisted Intervention – MICCAI 2020. Lecture Notes in Computer Science. Springer; 2020:320–329. doi:10.1007/978-3-030-59725-2_31.",
 'wen2022': "Wen D, Khan SM, Ji Xu A, et al. Characteristics of publicly available skin cancer image datasets: a systematic review. Lancet Digit Health. 2022;4:e64–e74. doi:10.1016/S2589-7500(21)00252-1.",
 'adamson2018': "Adamson AS, Smith A. Machine learning and health care disparities in dermatology. JAMA Dermatol. 2018;154:1247. doi:10.1001/jamadermatol.2018.2348.",
 'morenotorres2012': "Moreno-Torres JG, Raeder T, Alaíz-Rodríguez R, Chawla NV, Herrera F. A unifying view on dataset shift in classification. Pattern Recognit. 2012;45:521–530. doi:10.1016/j.patcog.2011.06.019.",
 'saerens2002': "Saerens M, Latinne P, Decaestecker C. Adjusting the outputs of a classifier to new a priori probabilities: a simple procedure. Neural Comput. 2002;14:21–41. doi:10.1162/089976602753284446.",
 'henderson2018': "Henderson P, Islam R, Bachman P, Pineau J, Precup D, Meger D. Deep reinforcement learning that matters. Proc AAAI. 2018;32. doi:10.1609/aaai.v32i1.11694.",
 'bouthillier2021': "Bouthillier X, Delaunay P, Bronzi M, et al. Accounting for variance in machine learning benchmarks. Proc Mach Learn Syst (MLSys). 2021;3.",
 'holm1979': "Holm S. A simple sequentially rejective multiple test procedure. Scand J Stat. 1979;6:65–70.",
 'kruskal1952': "Kruskal WH, Wallis WA. Use of ranks in one-criterion variance analysis. J Am Stat Assoc. 1952;47:583–621. doi:10.1080/01621459.1952.10483441.",
 'mann1947': "Mann HB, Whitney DR. On a test of whether one of two random variables is stochastically larger than the other. Ann Math Stat. 1947;18:50–60. doi:10.1214/aoms/1177730491.",
 'kerby2014': "Kerby DS. The simple difference formula: an approach to teaching nonparametric correlation. Compr Psychol. 2014;3:11.IT.3.1. doi:10.2466/11.IT.3.1.",
 'efron1987': "Efron B. Better bootstrap confidence intervals. J Am Stat Assoc. 1987;82:171–185. doi:10.1080/01621459.1987.10478410.",
 'clopper1934': "Clopper CJ, Pearson ES. The use of confidence or fiducial limits illustrated in the case of the binomial. Biometrika. 1934;26:404–413. doi:10.1093/biomet/26.4.404.",
 'cleveland1979': "Cleveland WS. Robust locally weighted regression and smoothing scatterplots. J Am Stat Assoc. 1979;74:829–836. doi:10.1080/01621459.1979.10481038.",
 'kelley1935': "Kelley TL. An unbiased correlation ratio measure. Proc Natl Acad Sci USA. 1935;21:554–559. doi:10.1073/pnas.21.9.554.",
 'cohen1960': "Cohen J. A coefficient of agreement for nominal scales. Educ Psychol Meas. 1960;20:37–46. doi:10.1177/001316446002000104.",
 'chicco2020': "Chicco D, Jurman G. The advantages of the Matthews correlation coefficient (MCC) over F1 score and accuracy in binary classification evaluation. BMC Genomics. 2020;21:6. doi:10.1186/s12864-019-6413-7.",
 'saito2015': "Saito T, Rehmsmeier M. The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. PLoS One. 2015;10:e0118432. doi:10.1371/journal.pone.0118432.",
 'mongan2020': "Mongan J, Moy L, Kahn CE Jr. Checklist for artificial intelligence in medical imaging (CLAIM): a guide for authors and reviewers. Radiol Artif Intell. 2020;2:e200029. doi:10.1148/ryai.2020200029.",
 'collins2024': "Collins GS, Moons KGM, Dhiman P, et al. TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods. BMJ. 2024;385:e078378. doi:10.1136/bmj-2023-078378.",
 'lekadir2025': "Lekadir K, Frangi AF, Porras AR, et al. FUTURE-AI: international consensus guideline for trustworthy and deployable artificial intelligence in healthcare. BMJ. 2025;388:e081554. doi:10.1136/bmj-2024-081554.",
 'nagendran2020': "Nagendran M, Chen Y, Lovejoy CA, et al. Artificial intelligence versus clinicians: systematic review of design, reporting standards, and claims of deep learning studies. BMJ. 2020;368:m689. doi:10.1136/bmj.m689.",
 'kelly2019': "Kelly CJ, Karthikesalingam A, Suleyman M, Corrado G, King D. Key challenges for delivering clinical impact with artificial intelligence. BMC Med. 2019;17. doi:10.1186/s12916-019-1426-2.",
 'kassem2021': "Kassem MA, Hosny KM, Damaševičius R, Eltoukhy MM. Machine learning and deep learning methods for skin lesion classification and diagnosis: a systematic review. Diagnostics. 2021;11:1390. doi:10.3390/diagnostics11081390.",
 'lipton2018': "Lipton ZC, Wang YX, Smola A. Detecting and correcting for label shift with black box predictors. Proc ICML. 2018;80.",
 # --- added for the resubmission literature review / SOTA comparison / evaluation-integrity discussion ---
 'jamaludin2026': "Jamaludin RAM, Kim CS. Evaluation integrity in hybrid quantum-classical skin lesion classification: a data-leakage audit and lesion-level correction on HAM10000. IEEE Access. 2026;14:101415–101428. doi:10.1109/ACCESS.2026.3707673.",
 'abhishek2025': "Abhishek K, Jain A, Hamarneh G. Investigating the quality of DermaMNIST and Fitzpatrick17k dermatological image datasets. Sci Data. 2025;12. doi:10.1038/s41597-025-04382-5.",
 'combalia2022': "Combalia M, Codella N, Rotemberg V, et al. Validation of artificial intelligence prediction models for skin cancer diagnosis using dermoscopy images: the 2019 International Skin Imaging Collaboration Grand Challenge. Lancet Digit Health. 2022;4:e330–e339. doi:10.1016/S2589-7500(22)00021-8.",
 'yu2022ext': "Yu AC, Mohajer B, Eng J. External validation of deep learning algorithms for radiologic diagnosis: a systematic review. Radiol Artif Intell. 2022;4. doi:10.1148/ryai.210064.",
 'arun2021': "Arun N, Gaw N, Singh P, et al. Assessing the trustworthiness of saliency maps for localizing abnormalities in medical imaging. Radiol Artif Intell. 2021;3. doi:10.1148/ryai.2021200267.",
 'saporta2022': "Saporta A, Gui X, Agrawal A, et al. Benchmarking saliency methods for chest X-ray interpretation. Nat Mach Intell. 2022;4:867–878. doi:10.1038/s42256-022-00536-x.",
 'gomez2022': "Gomez T, Fréour T, Mouchère H. Metrics for saliency map evaluation of deep learning explanation methods. In: Pattern Recognition and Artificial Intelligence (ICPRAI 2022), Lecture Notes in Computer Science. Springer; 2022:84–95. doi:10.1007/978-3-031-09037-0_8.",
 'fogelberg2023': "Fogelberg K, Chamarthi S, Maron RC, et al. Domain shifts in dermoscopic skin cancer datasets: evaluation of essential limitations for clinical translation. New Biotechnol. 2023;76:106–117. doi:10.1016/j.nbt.2023.04.006.",
 'matta2024': "Matta S, Lamard M, Zhang P, et al. A systematic review of generalization research in medical image classification. Comput Biol Med. 2024;183:109256. doi:10.1016/j.compbiomed.2024.109256.",
 'dagoglu2025': "Dağoğlu Hark B. Beyond accuracy: methodological evaluation of performance stability in deep learning models on the HAM10000 multi-class skin lesion benchmark. Turkiye Klinikleri J Biostat. 2025;17:140–150. doi:10.5336/biostatic.2025-116893.",
 'giavinabianchi2023': "Giavina-Bianchi M, Vitor WG, Fornasiero de Paiva V, et al. Explainability agreement between dermatologists and five visual explanation techniques in deep neural networks for melanoma AI classification. Front Med. 2023;10. doi:10.3389/fmed.2023.1241484.",
 'fernandez2025': "Fernandez R, Guzmán-Ponce A, Fernandez-Beltran R, et al. Symmetry in explainable AI: a morphometric deep learning analysis for skin lesion classification. Symmetry. 2025;17:1264. doi:10.3390/sym17081264.",
 'sonune2026': "Sonune BS, Udaykumar R, Nemane SG, et al. Interpretable deep learning in dermoscopy: an XAI-driven ensemble for skin lesion diagnosis. IEEE Access. 2026;14:62126–62142. doi:10.1109/ACCESS.2026.3684257.",
 'hosny2024': "Hosny KM, Said W, Elmezain M, et al. Explainable deep inherent learning for multi-classes skin lesion classification. Appl Soft Comput. 2024;159:111624. doi:10.1016/j.asoc.2024.111624.",
 'picard2021': "Picard D. Torch.manual_seed(3407) is all you need: on the influence of random seeds in deep learning architectures for computer vision. arXiv:2109.08203. 2021.",
 'tampu2022': "Tampu IE, Eklund A, Haj-Hosseini N. Inflation of test accuracy due to data leakage in deep learning-based classification of OCT images. Sci Data. 2022;9. doi:10.1038/s41597-022-01618-6.",
 'khan2026': "Khan MAA, Tahir M, Irtaza SA, Mughal MR. SynthraXCoreNet: an interpretable, well-calibrated six-CNN ensemble for dermoscopic skin-lesion classification. IEEE Access. 2026;14:6287–6309. doi:10.1109/ACCESS.2025.3641868.",
 'murali2026': "Murali P, Mazumder DH. DermaScanAI: an explainable hybrid deep learning framework for automated skin lesion classification using dual attention and metadata fusion. Sci Rep. 2026;16. doi:10.1038/s41598-026-46011-0.",
 'sekaran2026': "Sekaran R, Gopu A, A Sk, Reddy SS. Shift-consistent and explanation-stable distributed hybrid quantum learning for domain-generalized skin lesion classification. Sci Rep. 2026. doi:10.1038/s41598-026-66784-8.",
 'cino2025': "Cino L, Distante C, Martella A, Mazzeo PL. Skin lesion classification through test time augmentation and explainable artificial intelligence. J Imaging. 2025;11:15. doi:10.3390/jimaging11010015.",
 'he2016': "He K, Zhang X, Ren S, Sun J. Deep residual learning for image recognition. In: Proc IEEE Conf Comput Vis Pattern Recognit (CVPR). 2016:770–778. doi:10.1109/CVPR.2016.90.",
 'jacob2018': "Jacob B, Kligys S, Chen B, et al. Quantization and training of neural networks for efficient integer-arithmetic-only inference. In: Proc IEEE/CVF Conf Comput Vis Pattern Recognit (CVPR). 2018:2704–2713. doi:10.1109/CVPR.2018.00286.",
 'krishnamoorthi2018': "Krishnamoorthi R. Quantizing deep convolutional networks for efficient inference: a whitepaper. arXiv:1806.08342. 2018.",
 'nagel2021': "Nagel M, Fournarakis M, Amjad RA, et al. A white paper on neural network quantization. arXiv:2106.08295. 2021.",
 'rokh2023': "Rokh B, Azarpeyvand A, Khanteymoori A. A comprehensive survey on model quantization for deep neural networks in image classification. ACM Trans Intell Syst Technol. 2023;14:1–50. doi:10.1145/3623402.",
 'kerkouri2024': "Kerkouri MA, Tliba M, Chetouani A, Bruno A. Quantization effects on neural networks perception: how would quantization change the perceptual field of vision models? In: 2024 IEEE 13th Int Conf Image Process Theory Tools Appl (IPTA). 2024:1–6. doi:10.1109/IPTA62886.2024.10755934.",
 'rabbi2026': "Rabbi KK, Farabe MZAZ, Rahman MS. Same predictions, different reasons: the effect of quantization on model explanations. arXiv:2607.22872. 2026.",
}


# Stable labels for every table and figure, matched on the first words of the caption. Prose cites them as [[T:label]] / [[F:label]]
# (ranges [[T:a..b]], lists [[T:a,b]]), so numbering follows the order in which items are added and can never go stale.
CAPTION_LABELS = [
    ('Dataset partition', 'tab:dataset'), ('Held-out test performance', 'tab:perf_seed'), ('Per-class test performance', 'tab:perclass'),
    ('Calibration before and after', 'tab:calib'), ('Convergence audit', 'tab:converge'), ('Explanation and confidence measures', 'tab:correct_wrong'),
    ('Class-wise LSAS', 'tab:classwise'), ('Model-independent simulation', 'tab:simulation'), ('Lesion-size audit for all 990', 'tab:size_audit'),
    ('Deletion and insertion AUC', 'tab:faith_auc'), ('Faithfulness gains', 'tab:faith_gains'), ('Skin-tone-proxy analysis', 'tab:skin'),
    ('Stepwise decomposition', 'tab:pad_decomp'), ('Adaptation ladder', 'tab:pad_ladder'), ('PAD-UFES-20 accuracy by recorded', 'tab:pad_fitz'),
    ('Recent skin-lesion classification', 'tab:sota'), ('Null-controlled excess insertion gain', 'tab:null_faith'),
    ('Sensitivity of excess insertion/deletion', 'tab:sens'),
    ('Consolidated architecture and workflow', 'fig:pipeline'), ('Training curves', 'fig:training'), ('Per-class recall and F1', 'fig:perf_calib'),
    ('Distributions of LSAS, excess LSAS', 'fig:correct_wrong'), ('Examples (reference seed)', 'fig:gallery'), ('Lesion-size audit (reference seed)', 'fig:size_audit'),
    ('Deletion and insertion curves', 'fig:faith'), ('Stochastic saliency stability', 'fig:stability'), ('ITA distribution', 'fig:skin'),
    ('Left: stepwise accuracy decomposition', 'fig:pad'), ('Null-controlled faithfulness (reference', 'fig:null_faith'),
]


def _std_caption(c):
    c = c.strip()
    c = c[0].upper() + c[1:]
    return c if c.endswith(('.', '?', '!')) else c + '.'


class Manuscript:
    def __init__(self):
        self.items = []; self.order = []; self.fig_n = 0; self.tab_n = 0; self.labels = {}; self.used_labels = set()

    # ------------------------------------------------------------ cross-references
    def _register(self, kind, caption, label, n):
        if label is None:
            for pre, lab in CAPTION_LABELS:
                if lab.startswith(kind + ':') and caption.startswith(pre): label = lab; break
        assert label is not None, f'no label for {kind} caption: {caption[:60]}'
        assert label not in self.labels, f'duplicate label {label}'
        self.labels[label] = (kind, n)

    def _num(self, lab):
        assert lab in self.labels, f'unknown cross-reference label {lab}'
        return self.labels[lab][1]

    def res(self, text):
        """Resolve [[T:..]] / [[F:..]] placeholders into 'Table 3', 'Tables 2–3', 'Figure 6', ..."""
        def rep(m):
            kind, body = m.group(1), m.group(2); word = 'Table' if kind == 'T' else 'Figure'
            labs = []
            pre = 'tab:' if kind == 'T' else 'fig:'; fq = lambda x: x.strip() if ':' in x else pre + x.strip()
            for part in body.split(','):
                part = fq(part)
                if '..' in part:
                    a, b = [fq(x) for x in part.replace(pre, '').split('..')]; lo, hi = self._num(a), self._num(b); nums = list(range(lo, hi + 1))
                    for lab in (a, b): self.used_labels.add(lab)
                    labs += nums
                else:
                    self.used_labels.add(part); labs.append(self._num(part))
            nums = sorted(set(labs))
            if len(nums) == 1: return f'{word} {nums[0]}'
            runs = []; i = 0
            while i < len(nums):
                j = i
                while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1: j += 1
                runs.append(str(nums[i]) if i == j else (f'{nums[i]} and {nums[j]}' if j == i + 1 and len(nums) == 2 else f'{nums[i]}–{nums[j]}')); i = j + 1
            return f'{word}s ' + (' and '.join(runs) if len(runs) == 2 else ', '.join(runs))
        text = re.sub(r'\[\[([TF]):([^\]]+)\]\]', rep, text)
        text = re.sub(r'(Sections? \d+(?:\.\d+)?)-(\d+(?:\.\d+)?)', lambda m: m.group(1) + '\u2013' + m.group(2), text)   # uniform en dash in section ranges
        # uniform dispersion notation: no plus/minus sign anywhere ('0.808 (SD 0.006)', 'mean and SD')
        text = re.sub(r'\(\u00b1([\d.\u2013-]+)\)', r'(SD \1)', text)
        text = re.sub(r'(\d+(?:[.,]\d+)*%?)\s*\u00b1\s*(\d+(?:[.,]\d+)*%?)', r'\1 (SD \2)', text)
        text = re.sub(r'\u00b1\s*(?=\d)', '', text)   # bare half-widths (e.g. bootstrap interval half-width)
        text = text.replace('mean \u00b1 SD', 'mean and SD').replace('Mean \u00b1 SD', 'Mean and SD').replace('mean +/- SD', 'mean and SD')
        return text

    def check_xrefs(self):
        """Returns a list of problems: unresolved placeholders, tables/figures never cited in running text."""
        probs = []
        for it in self.items:
            strs = [it[1]] if it[0] == 'p' else (it[1] if it[0] == 'ul' else ([str(c) for r in it[2] for c in r] + ([it[4]] if it[4] else []) if it[0] == 't' else []))
            for s in strs:
                r = self.res(s)
                if '[[' in r: probs.append('unresolved placeholder: ' + r[:80])
        cited = set(self.used_labels)
        for lab in self.labels:
            if lab not in cited: probs.append(f'{lab} (Table/Figure {self.labels[lab][1]}) is never cited in the text')
        return probs

    def cite(self, text):
        def rep(m):
            keys = [k.strip() for k in m.group(1).split(',')]; nums = []
            for k in keys:
                assert k in REFS, f'unknown ref key {k}'
                if k not in self.order: self.order.append(k)
                nums.append(self.order.index(k) + 1)
            nums = sorted(set(nums)); out = []; i = 0
            while i < len(nums):
                j = i
                while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1: j += 1
                out.append(str(nums[i]) if i == j else (f'{nums[i]},{nums[j]}' if j == i + 1 else f'{nums[i]}–{nums[j]}')); i = j + 1
            return '[' + ','.join(out) + ']'
        return re.sub(r'\[@([^\]]+)\]', rep, text)

    def h(self, level, text): self.items.append(('h', level, text))
    def p(self, text, style=None): self.items.append(('p', self.cite(text), style))
    def bullets(self, lst): self.items.append(('ul', [self.cite(t) for t in lst]))
    def eq(self, text, num): self.items.append(('eq', text, num))
    def table(self, header, rows, caption, note=None, widths=None, label=None):
        def c1(v): return self.cite(v) if isinstance(v, str) else v
        header = [c1(h) for h in header]; rows = [[c1(v) for v in r] for r in rows]
        caption = _std_caption(caption); self.tab_n += 1; self._register('tab', caption, label, self.tab_n)
        if note and not note.lstrip('*').lstrip().startswith(('Note', 'Notes')): note = 'Notes: ' + note
        self.items.append(('t', header, rows, f'Table {self.tab_n}. {self.cite(caption)}', self.cite(note) if note else None, widths)); return self.tab_n
    def fig(self, path, caption, width_cm=16.0, label=None):
        caption = _std_caption(caption); self.fig_n += 1; self._register('fig', caption, label, self.fig_n)
        self.items.append(('f', str(path), f'Figure {self.fig_n}. {self.cite(caption)}', width_cm)); return self.fig_n

    # ---------------------------------------------------------------- writers
    def refs_list(self): return [f'{i + 1}. {REFS[k]}' for i, k in enumerate(self.order)]

    def to_markdown(self):
        out = []; R = self.res
        for it in self.items:
            t = it[0]
            if t == 'h': out.append('\n' + '#' * it[1] + ' ' + R(it[2]) + '\n')
            elif t == 'p': out.append(R(it[1]) + '\n')
            elif t == 'ul': out += ['- ' + R(x) for x in it[1]] + ['']
            elif t == 'eq': out.append(f'    {it[1]}    ({it[2]})\n')
            elif t == 't':
                out.append('**' + R(it[3]) + '**\n'); out.append('| ' + ' | '.join(R(str(h)) for h in it[1]) + ' |'); out.append('|' + '---|' * len(it[1]))
                out += ['| ' + ' | '.join(R(str(c)) for c in r) + ' |' for r in it[2]]; out.append('\n' + (R(it[4]) + '\n' if it[4] else ''))
            elif t == 'f': out.append(f'![{R(it[2])}]({it[1]})\n\n*{R(it[2])}*\n')
        return '\n'.join(out)

    def to_docx(self, path):
        doc = Document(); sec = doc.sections[0]; sec.left_margin = sec.right_margin = Cm(2.0); sec.top_margin = sec.bottom_margin = Cm(2.0)
        st = doc.styles['Normal']; st.font.name = 'Times New Roman'; st.font.size = Pt(10.5); st.element.rPr.rFonts.set(qn('w:eastAsia'), 'Times New Roman')
        st.paragraph_format.space_after = Pt(4); st.paragraph_format.line_spacing = 1.08
        for lvl, sz in [(1, 14), (2, 12), (3, 11)]:
            hs = doc.styles[f'Heading {lvl}']; hs.font.name = 'Times New Roman'; hs.font.size = Pt(sz); hs.font.bold = True; hs.font.color.rgb = RGBColor(0, 0, 0)
            hs.element.rPr.rFonts.set(qn('w:eastAsia'), 'Times New Roman'); hs.paragraph_format.space_before = Pt(10 if lvl == 1 else 7); hs.paragraph_format.space_after = Pt(3)

        R = self.res

        def add_runs(par, text):
            text = R(text)
            for part in re.split(r'(\*\*[^*]+\*\*|__[^_]+__)', text):
                if part.startswith('**'): r = par.add_run(part[2:-2]); r.bold = True
                elif part.startswith('__'): r = par.add_run(part[2:-2]); r.italic = True
                else: par.add_run(part)

        def shade(cell, color):
            tcPr = cell._tc.get_or_add_tcPr(); shd = OxmlElement('w:shd'); shd.set(qn('w:val'), 'clear'); shd.set(qn('w:color'), 'auto'); shd.set(qn('w:fill'), color); tcPr.append(shd)

        for it in self.items:
            t = it[0]
            if t == 'h':
                if it[1] == 0:
                    par = doc.add_paragraph(); par.alignment = WD_ALIGN_PARAGRAPH.CENTER; r = par.add_run(R(it[2])); r.bold = True; r.font.size = Pt(16)
                else: doc.add_heading(R(it[2]), level=it[1])
            elif t == 'p':
                par = doc.add_paragraph(); add_runs(par, it[1])
                if it[2] == 'center': par.alignment = WD_ALIGN_PARAGRAPH.CENTER
                else: par.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            elif t == 'ul':
                for x in it[1]: par = doc.add_paragraph(style='List Bullet'); add_runs(par, x)
            elif t == 'eq':
                par = doc.add_paragraph(); par.alignment = WD_ALIGN_PARAGRAPH.CENTER; r = par.add_run(f'{it[1]}        ({it[2]})'); r.italic = True
            elif t == 't':
                cap = doc.add_paragraph(); r = cap.add_run(R(it[3])); r.bold = True; r.font.size = Pt(9.5); cap.paragraph_format.keep_with_next = True
                tb = doc.add_table(rows=1, cols=len(it[1])); tb.style = 'Table Grid'; tb.alignment = WD_TABLE_ALIGNMENT.CENTER
                for i, c in enumerate(it[1]):
                    cell = tb.rows[0].cells[i]; cell.text = ''; rr = cell.paragraphs[0].add_run(R(str(c))); rr.bold = True; rr.font.size = Pt(8.5); shade(cell, 'D9E2F3')
                for row in it[2]:
                    cells = tb.add_row().cells
                    for i, c in enumerate(row):
                        cells[i].text = ''; s = R(str(c)); bold = s.startswith('**'); rr = cells[i].paragraphs[0].add_run(s.strip('*')); rr.font.size = Pt(8.5); rr.bold = bold
                if it[4]: np_ = doc.add_paragraph(); rn = np_.add_run(R(it[4])); rn.italic = True; rn.font.size = Pt(8.5)
                else: doc.add_paragraph()
            elif t == 'f':
                doc.add_picture(it[1], width=Cm(it[3])); doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
                cap = doc.add_paragraph(); r = cap.add_run(R(it[2])); r.font.size = Pt(9); cap.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        doc.save(path)
