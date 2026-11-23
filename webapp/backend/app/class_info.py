"""
Class metadata, educational clinical descriptions, and risk context for HAM10000 classes.
Complies with clinical guidelines: Softmax scores are not diagnostic probabilities;
disclaimers must be presented on all outputs.
"""

CLASS_TO_IDX = {
    'akiec': 0,
    'bcc': 1,
    'bkl': 2,
    'df': 3,
    'mel': 4,
    'nv': 5,
    'vasc': 6
}

IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}

CLASS_DETAILS = {
    'akiec': {
        'code': 'akiec',
        'name': 'Actinic Keratoses and Intraepithelial Carcinoma',
        'short_name': 'Actinic Keratoses (AKIEC)',
        'type': 'Pre-malignant / In-situ Carcinoma',
        'risk_level': 'Moderate Risk',
        'risk_badge': 'warning',
        'description': (
            'Actinic keratoses (solar keratoses) and intraepithelial carcinomas (Bowen disease) '
            'are superficial cutaneous lesions induced by cumulative UV exposure. Although non-invasive, '
            'they carry a known risk of progression to invasive squamous cell carcinoma (SCC) if left untreated.'
        ),
        'clinical_guidance': (
            'Clinical evaluation by a dermatologist is recommended for possible topical therapies, '
            'cryotherapy, or excision. Sun protection measures should be reviewed.'
        ),
        'color': '#F59E0B'
    },
    'bcc': {
        'code': 'bcc',
        'name': 'Basal Cell Carcinoma',
        'short_name': 'Basal Cell Carcinoma (BCC)',
        'type': 'Malignant (Non-Melanoma Skin Cancer)',
        'risk_level': 'High Risk',
        'risk_badge': 'danger',
        'description': (
            'Basal cell carcinoma is the most prevalent form of skin cancer globally. Characterized by locally '
            'invasive behavior and slow growth, it rarely metastasizes but can cause substantial local tissue destruction.'
        ),
        'clinical_guidance': (
            'Requires formal medical diagnosis and dermatological consultation. Treatment options typically include '
            'surgical excision, Mohs micrographic surgery, or targeted topical/destructive modalities.'
        ),
        'color': '#EF4444'
    },
    'bkl': {
        'code': 'bkl',
        'name': 'Benign Keratosis-like Lesions',
        'short_name': 'Benign Keratosis (BKL)',
        'type': 'Benign',
        'risk_level': 'Low Risk',
        'risk_badge': 'success',
        'description': (
            'Encompasses common benign skin lesions including seborrheic keratoses, solar lentigines, '
            'and lichen planus-like keratoses (LPLK). These lesions are non-cancerous and do not undergo malignant degeneration.'
        ),
        'clinical_guidance': (
            'Typically no treatment is required unless irritated or for cosmetic reasons. Any rapidly changing lesion '
            'should still be inspected clinically.'
        ),
        'color': '#10B981'
    },
    'df': {
        'code': 'df',
        'name': 'Dermatofibroma',
        'short_name': 'Dermatofibroma (DF)',
        'type': 'Benign Fibrohistiocytic',
        'risk_level': 'Low Risk',
        'risk_badge': 'success',
        'description': (
            'A common benign cutaneous nodule commonly occurring on extremities, often exhibiting a positive '
            '"pinch" or dimple sign upon lateral compression.'
        ),
        'clinical_guidance': (
            'Benign condition requiring no routine excision unless symptomatic, bleeding, or clinically uncertain.'
        ),
        'color': '#06B6D4'
    },
    'mel': {
        'code': 'mel',
        'name': 'Melanoma',
        'short_name': 'Melanoma (MEL)',
        'type': 'Malignant Melanocytic Neoplasm',
        'risk_level': 'High Risk',
        'risk_badge': 'danger',
        'description': (
            'Melanoma is the most aggressive form of skin cancer, arising from melanocytes. Early identification '
            'and prompt surgical excision are critical to clinical prognosis.'
        ),
        'clinical_guidance': (
            'Urgent dermatological assessment and histological biopsy confirmation required. Avoid delay.'
        ),
        'color': '#DC2626'
    },
    'nv': {
        'code': 'nv',
        'name': 'Melanocytic Nevus (Common Mole)',
        'short_name': 'Melanocytic Nevi (NV)',
        'type': 'Benign Melanocytic',
        'risk_level': 'Low Risk',
        'risk_badge': 'success',
        'description': (
            'Common benign pigmented neoplasms formed by proliferation of melanocytes (moles). The vast majority '
            'remain benign throughout a patient’s lifetime.'
        ),
        'clinical_guidance': (
            'Routine monitoring using the ABCDE rule is advisable. If changes in asymmetry, border, color, or diameter occur, '
            'consult a dermatologist.'
        ),
        'color': '#3B82F6'
    },
    'vasc': {
        'code': 'vasc',
        'name': 'Vascular Lesions',
        'short_name': 'Vascular Lesions (VASC)',
        'type': 'Benign Vascular',
        'risk_level': 'Low Risk',
        'risk_badge': 'success',
        'description': (
            'Encompasses cherry angiomas, pyogenic granulomas, angiokeratomas, and vascular malformations. '
            'Mostly benign proliferations of blood vessels.'
        ),
        'clinical_guidance': (
            'Benign nature in most cases. Evaluation recommended if recurrent ulceration, unexplained bleeding, or rapid enlargement occurs.'
        ),
        'color': '#8B5CF6'
    }
}

MODEL_EVALUATION_METRICS = {
    'backbone': 'ConvNeXt-Tiny (convnext_tiny)',
    'dataset': 'HAM10000 (ViDIR Group, Medical University of Vienna)',
    'input_resolution': '224 x 224 RGB',
    'best_epoch': 20,
    'total_epochs': 25,
    'test_accuracy': 83.46,
    'balanced_accuracy': 81.23,
    'macro_f1': 77.83,
    'macro_auc': 94.76,
    'per_class_recall': {
        'akiec': 60.98,
        'bcc': 88.89,
        'bkl': 76.36,
        'df': 90.00,
        'mel': 71.43,
        'nv': 87.22,
        'vasc': 93.75
    }
}

MEDICAL_DISCLAIMER = (
    "This application is an educational and research-based AI skin lesion classification tool. "
    "Its predictions are generated by a deep-learning model trained on the HAM10000 dataset. "
    "The results are not a medical diagnosis and should not replace consultation with a qualified "
    "dermatologist or healthcare professional. The model may produce incorrect predictions, and its "
    "performance on real-world images may differ from its test performance."
)
