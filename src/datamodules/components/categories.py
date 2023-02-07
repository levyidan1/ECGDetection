from typing import Dict, List, Iterator, Any


class Categories:
    """
    This class is used to store the categories of the dataset.
    It maps the categories from full names to the corresponding names used in the dataset.
    Example category mapping:
    categories_lookup_dict = {
            'Atrial fibrillation': ['Atrial fibrillation'],
            'Atrial flutter': ['Atrial flutter'],
            'Atrial premature complex(es) - APC APB': ['Atrial premature complex(es)',
                                                       'Atrial premature complexes_ nonconducted'],
                                                       }
    """
    def __init__(self, categories_lookup_dict: Dict[str, List[str]]):
        self.categories_lookup_dict = categories_lookup_dict
        self.categories = sorted(categories_lookup_dict.keys())
        self.num_categories = len(self.categories)
        self.category_to_idx = {cat: idx for idx, cat in enumerate(self.categories)}

    def __len__(self) -> int:
        return self.num_categories

    def __getitem__(self, idx: int) -> str:
        return self.categories[idx]

    def __iter__(self) -> Iterator[str]:
        return iter(self.categories)

    def __contains__(self, item: str) -> bool:
        return item in self.categories

    def __repr__(self) -> str:
        return f"Categories: {self.categories}"

    def __eq__(self, other: Any) -> bool:
        return (
            self.categories_lookup_dict == other.categories_lookup_dict
            if isinstance(other, Categories)
            else False
        )

    def __hash__(self) -> int:
        return hash(self.categories_lookup_dict)

    def get_category_idx(self, category: str) -> int:
        """
        Get the index of a category.
        Args:
            category: the category to get the index of.
        Returns:
            The index of the category.
        """
        return self.category_to_idx[category]

    def get_category(self, idx: int) -> str:
        """
        Get the category of an index.
        Args:
            idx: the index to get the category of.
        Returns:
            The category of the index.
        """
        return self.categories[idx]

    def get_categories(self, idxs: List[int]) -> List[str]:
        """
        Get the categories of a list of indices.
        Args:
            idxs: the indices to get the categories of.
        Returns:
            The categories of the indices.
        """
        return [self.categories[idx] for idx in idxs]

    def get_categories_lookup_dict(self) -> Dict[str, List[str]]:
        """
        Get the category mapping.
        Returns:
            The category mapping.
        """
        return self.categories_lookup_dict

    def get_categories_lookup_dict_inverse(self) -> Dict[str, str]:
        """
        Get the inverse category mapping.
        Returns:
            The inverse category mapping.
        """
        return {cat: cat_full for cat_full, cats in self.categories_lookup_dict.items() for cat in cats}


ny_categories_lookup_dict = \
    {
        'AV Block - First-degree': ['AV Block', ' type I'],
        'AV Block - Second-degree': ['AV Block - Second-degree',
                                     ' Mobitz type II',
                                     ' Mobitz type I (Wenckebach)'],
        'AV Block - Third-degree (Complete)': ' complete (third-degree)',
        'Abnormal P-wave Axis': 'Abnormal P-wave axis',
        'Acute Pericarditis': 'Acute pericarditis',
        'Atrial Fibrillation': 'Atrial fibrillation',
        'Atrial Flutter': 'Atrial flutter',
        'Atrial Premature Complex(es) - APC APB': ['Atrial premature complex(es) - APC APB',
                                                   'Ectopic atrial rhythm'],
        'Complete Heart Block': 'complete (third-degree)',
        'Complete Left Bundle Branch Block': 'Bundle Branch Block - Left - LBBB',
        'Complete Right Bundle Branch Block': 'Bundle Branch Block - Right - RBBB',
        'Digitalis Effect': 'Digitalis effect',
        'Early Repolarization': 'Early repolarization',
        'Ectopic Atrial Tachycardia': 'Ectopic atrial tachycardia',
        'Electrode Reversal': 'Extremity electrode reversal',
        'Fusion Beats': 'Fusion complex(es)',
        'Incomplete Right Bundle Branch Block': 'Bundle-branch block - RBBB - incomplete',
        'Intraventricular Conduction Delay': 'Intraventricular conduction delay',
        'Junctional Escape': 'Junctional escape complex(es)',
        'Junctional Tachycardia': 'Junctional tachycardia',
        'Left Anterior Fascicular Block': 'Left anterior fascicular block',
        'Left Atrial Enlargement': 'Left atrial enlargement',
        'Left Axis Deviation': 'Left-axis deviation',
        'Left Posterior Fascicular Block': 'Left posterior fascicular block',
        'Left Ventricular Hypertrophy': 'Left ventricular hypertrophy',
        'Low QRS Voltages': 'Low voltage',
        'Myocardial Infarction': ['STEMI - Anteroseptal',
                                  'STEMI - Anterior',
                                  'STEMI - Inferior or Inferolateral',
                                  'STEMI - Lateral',
                                  'STEMI - Posterior',
                                  'STEMI - Right Ventricular'],
        'Myocardial Infarction - Anterior': 'STEMI - Anterior',
        'Myocardial Infarction - Anteroseptal': 'STEMI - Anteroseptal',
        'Myocardial Infarction - Inferior Or Inferolateral': 'STEMI - Inferior or Inferolateral',
        'Myocardial Infarction - Lateral': 'STEMI - Lateral',
        'Myocardial Infarction - Posterior': 'STEMI - Posterior',
        'Myocardial Infarction - Right Ventricular': 'STEMI - Right Ventricular',
        'Nonspecific Intraventricular Conduction Disorder': 'Intraventricular conduction delay',
        'Normal Sinus Rhythm': 'Normal ECG',
        'Normal Variant': 'Normal variant',
        'PR Interval - Prolonged': 'PR Interval - Prolonged',
        'PR Interval - Short': 'PR Interval - Short',
        'Pacing': ['Pacing - Ventricular-paced complex(es) or rhythm',
                   'Pacing - Atrial-sensed ventricular-paced complex(es) or rhythm',
                   'Pacing - AV dual-paced complex(es) or rhythm',
                   'Pacing - Atrial-paced complex(es) or rhythm',
                   'Pacing Ventricular pacing',
                   'Pacing - Failure to pace',
                   'Pacing - Failure to inhibit',
                   'Pacing - Failure to capture'],
        'Premature Ventricular Contractions': 'Ventricular premature complex(es) - VPB - VPC',
        'Pulmonary Disease': 'Pulmonary disease',
        'QT Interval - Prolonged': 'QT Interval - Prolonged',
        'QT Interval - Short': 'QT Interval - Short',
        'Right Atrial Enlargement': 'Right atrial enlargement',
        'Right Axis Deviation': 'Right-axis deviation',
        'Right Superior Axis': 'Right superior axis',
        'Right Ventricular Hypertrophy': 'Right ventricular hypertrophy',
        'ST Changes': ['ST changes - Nonspecific T-wave abnormality',
                       'ST changes - Nonspecific ST deviation',
                       'ST changes - Nonspecific ST deviation with T-wave change',
                       'ST-T change due to ventricular hypertrophy'],
        'ST Changes - Nonspecific ST Deviation': 'ST changes - Nonspecific ST deviation',
        'ST Changes - Nonspecific ST Deviation With T-wave Change': 'ST changes - Nonspecific ST deviation with T-wave change',
        'ST Changes - Nonspecific T-wave Abnormality': 'ST changes - Nonspecific T-wave abnormality',
        'Sinosatrial Block': 'Sinosatrial block',
        'Sinus Arrhythmia': 'Sinus arrhythmia',
        'Sinus Bradycardia': 'Sinus bradycardia',
        'Sinus Tachycardia': 'Sinus tachycardia',
        'Supraventricular Tachycardia': 'Supraventricular tachycardia',
        'T Wave Abnormal': ['ST-T change due to ventricular hypertrophy',
                            'ST changes - Nonspecific T-wave abnormality',
                            'ST changes - Nonspecific ST deviation with T-wave change'],
        'TU Fusion': 'TU fusion',
        'Ventricular Escape Rhythm': 'Ventricular escape complex(es)',
        'Ventricular Pre Excitation': 'Ventricular preexcitation',
        'Ventricular Tachycardia': 'Ventricular tachycardia',
        'Wandering Atrial Pacemaker': 'Wandering atrial pacemaker',
        'Wolf-Parkinson-White': 'Wolff-Parkinson-White'
    }

br_categories_lookup_dict = {
    'AV Block - First-degree': '1dAVb',
    'Atrial Fibrillation': 'AF',
    'Complete Left Bundle Branch Block': 'LBBB',
    'Complete Right Bundle Branch Block': 'RBBB',
    'Sinus Bradycardia': 'SB',
    'Sinus Tachycardia': 'ST'
}

physionet_categories_lookup_dict = {
    'AV Block': ['AVB'],
    'AV Block - First-degree': ['IAVB'],
    'AV Block - Second-degree': ['IIAVB', 'IIAVBII', 'MoI'],
    'AV Dissociation': ['AVD'],
    'AV Junctional Rhythm': ['AVJR'],
    'AV Node Reentrant Tachycardia (AVNRT)': ['AVNRT', 'AVRT'],
    'Abnormal QRS': ['abQRS'],
    'Accelerated Atrial Escape Rhythm': ['AAR'],
    'Accelerated Idioventricular Rhythm': ['AIVR'],
    'Accelerated Junctional Rhythm': ['AJR'],
    'Anterior Ischemia': ['AnMIs'],
    'Atrial Bigeminy': ['AB', 'SVB'],
    'Atrial Escape Beat': ['AED'],
    'Atrial Fibrillation': ['AF', 'AFAFL', 'CAF', 'PAF', 'RAF'],
    'Atrial Flutter': ['AFL', 'AFAFL'],
    'Atrial Hypertrophy': ['AH'],
    'Atrial Premature Complex(es) - APC APB': ['PAC', 'BPAC'],
    'Atrial Rhythm': ['ARH', 'SAAWR'],
    'Atrial Tachycardia': ['ATach'],
    'Brugada Syndrome': ['BRU'],
    'Clockwise Or Counterclockwise Vectorcardiographic Loop': ['CVCL/CCVCL'],
    'Clockwise Rotation': ['CR'],
    'Complete Heart Block': ['CHB'],
    'Complete Left Bundle Branch Block': ['CLBBB'],
    'Complete Right Bundle Branch Block': ['CRBBB'],
    'Coronary Heart Disease': ['CHD'],
    'Countercolockwise Rotation': ['CCR'],
    'Diffuse Intraventricular Block': ['DIB'],
    'Early Repolarization': ['ERe'],
    'Electrode Reversal': ['ALR'],
    'FQRS Wave': ['FQRS'],
    'Fusion Beats': ['FB'],
    'Heart Failure': ['HF'],
    'Heart Valve Disorder': ['HVD'],
    'High T-voltage': ['HTV'],
    'Idioventricular Rhythm': ['IR'],
    'Incomplete Left Bundle Branch Block': ['ILBBB'],
    'Incomplete Right Bundle Branch Block': ['IRBBB'],
    'Inferior Ischaemia': ['IIs'],
    'Inferior ST Segment Depression': ['ISTD'],
    'Junctional Escape': ['JE'],
    'Junctional Premature Complex': ['JPC'],
    'Junctional Tachycardia': ['JTach'],
    'Lateral Ischemia': ['LIs'],
    'Left Anterior Fascicular Block': ['LAnFB'],
    'Left Atrial Enlargement': ['LAE'],
    'Left Atrial Hypertrophy': ['LAH'],
    'Left Axis Deviation': ['LAD'],
    'Left Posterior Fascicular Block': ['LPFB'],
    'Left Ventricular High Voltage': ['LVHV'],
    'Left Ventricular Hypertrophy': ['LVH'],
    'Left Ventricular Strain': ['LVS'],
    'Low QRS Voltages': ['LQRSV'],
    'Myocardial Infarction': ['MI', 'OldMI', 'AMI', 'AnMI'],
    'Myocardial Infarction - Anterior': ['AnMI'],
    'Myocardial Ischemia': ['MIs', 'AMIs', 'CMI'],
    'Nonspecific Intraventricular Conduction Disorder': ['NSIVCB'],
    'P Wave Changes': ['PWC'],
    'PR Interval - Prolonged': ['LPR'],
    'PR Interval - Short': ['SPRI'],
    'Pacing': ['PR', 'AP', 'VPP'],
    'Poor R Wave Progression': ['PRWP'],
    'Premature Ventricular Contractions': ['VEB', 'VPVC'],
    'Prolonged P Wave': ['PPW'],
    'Q Wave Abnormal': ['QAb'],
    'QT Interval - Prolonged': ['LQT'],
    'QT Interval - Short': ['SQT'],
    'R Wave Abnormal': ['RAb'],
    'Right Atrial Abnormality': ['RAAb'],
    'Right Atrial High Voltage': ['RAHV'],
    'Right Atrial Hypertrophy': ['RAH'],
    'Right Axis Deviation': ['RAD'],
    'Right Ventricular Hypertrophy': ['RVH'],
    'ST Changes': ['STC', 'NSSTTA'],
    'ST Changes - Nonspecific ST Deviation With T-wave Change': ['NSSTTA'],
    'ST Elevation': ['STE'],
    'ST Interval Abnormal': ['STIAb'],
    'Sinosatrial Block': ['SAB'],
    'Sinus Arrhythmia': ['SA'],
    'Sinus Bradycardia': ['SB', 'Brady'],
    'Sinus Node Dysfunction': ['SND'],
    'Sinus Tachycardia': ['STach'],
    'Supraventricular Tachycardia': ['SVT', 'PSVT'],
    'T Wave Abnormal': ['TAb'],
    'T Wave Inversion': ['TInv'],
    'U Wave Abnormal': ['UAb'],
    'Ventricular Bigeminy': ['VBig'],
    'Ventricular Escape Beat': ['VEsB', 'VEsR'],
    'Ventricular Fibrillation': ['VF'],
    'Ventricular Flutter': ['VFL'],
    'Ventricular Pre Excitation': ['VPEx'],
    'Ventricular Tachycardia': ['VTach', 'PVT'],
    'Ventricular Trigeminy': ['VTrig'],
    'Wandering Atrial Pacemaker': ['WAP'],
    'Wolff-Parkinson-White': ['WPW']
}

sph_categories_lookup_dict = {
    'AV Block - Complete': ['AV block_ complete (third-degree)'],
    'AV Block - First-degree': ['AV block_ varying conduction'],
    'AV Block - Second-degree': ['Second-degree AV block_ Mobitz type I (Wenckebach)',
                                 'Second-degree AV block_ Mobitz type II',
                                 'AV block_ advanced (high-grade)',
                                 '2:1 AV block'],
    'AV Conduction Ratio - N:D': ['AV conduction ratio N:D'],
    'Atrial Fibrillation': ['Atrial fibrillation'],
    'Atrial Flutter': ['Atrial flutter'],
    'Atrial Premature Complex(es) - APC APB': ['Atrial premature complex(es)',
                                               'Atrial premature complexes_ nonconducted'],
    'Complete Left Bundle Branch Block': ['Left bundle-branch block'],
    'Complete Right Bundle Branch Block': ['Right bundle-branch block'],
    'Early Repolarization': ['Early repolarization'],
    'Incomplete Right Bundle Branch Block': ['Incomplete right bundle-branch block'],
    'Junctional Premature Complex': ['Junctional premature complex(es)'],
    'Left Anterior Fascicular Block': ['Left anterior fascicular block'],
    'Left Atrial Enlargement': ['Left atrial enlargement'],
    'Left Axis Deviation': ['Left-axis deviation'],
    'Left Posterior Fascicular Block': ['Left posterior fascicular block'],
    'Left Ventricular Hypertrophy': ['Left ventricular hypertrophy'],
    'Low QRS Voltages': ['Low voltage'],
    'Myocardial Infarction': ['Anterior MI',
                              'Anteroseptal MI',
                              'Extensive anterior MI',
                              'Inferior MI'],
    'Myocardial Infarction - Anterior': ['Anterior MI', 'Extensive anterior MI'],
    'Myocardial Infarction - Anteroseptal': ['Anteroseptal MI'],
    'Myocardial Infarction - Inferior Or Inferolateral': ['Inferior MI'],
    'Normal Sinus Rhythm': ['Normal ECG'],
    'PR Interval - Prolonged': ['Prolonged PR interval'],
    'PR Interval - Short': ['Short PR interval'],
    'Premature Ventricular Contractions': ['Ventricular premature complex(es)'],
    'QT Interval - Prolonged': ['Prolonged QT interval'],
    'Right Axis Deviation': ['Right-axis deviation'],
    'Right Ventricular Hypertrophy': ['Right ventricular hypertrophy'],
    'ST Changes': ['ST deviation',
                   'ST deviation with T-wave change',
                   'ST-T change due to ventricular hypertrophy'],
    'ST Changes - Nonspecific ST Deviation': ['ST deviation'],
    'ST Changes - Nonspecific ST Deviation With T-wave Change': ['ST deviation with T-wave change'],
    'Sinus Arrhythmia': ['Sinus arrhythmia'],
    'Sinus Bradycardia': ['Sinus bradycardia'],
    'Sinus Tachycardia': ['Sinus tachycardia'],
    'T Wave Abnormal': ['T-wave abnormality'],
    'TU Fusion': ['TU fusion'],
    'Ventricular Pre Excitation': ['Ventricular preexcitation']

}

dataset_lookup_dicts = {
    'NY': Categories(ny_categories_lookup_dict),
    'Brazilian': Categories(br_categories_lookup_dict),
    'SPH': Categories(sph_categories_lookup_dict),
    'CPSC': Categories(physionet_categories_lookup_dict),
    'CPSC_Extra': Categories(physionet_categories_lookup_dict),
    'StPetersburg': Categories(physionet_categories_lookup_dict),
    'PTB': Categories(physionet_categories_lookup_dict),
    'PTB_XL': Categories(physionet_categories_lookup_dict),
    'Georgia': Categories(physionet_categories_lookup_dict),
    'Chapman_Shaoxing': Categories(physionet_categories_lookup_dict),
    'Ningbo': Categories(physionet_categories_lookup_dict)
}

