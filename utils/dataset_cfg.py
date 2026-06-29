"""
Dataset configuration classes for MAESTRO multimodal time series framework.

Each class specifies modalities, sampling rates, feature dimensions,
train/val/eval splits, and cross-validation folds.

Fold convention (Option B):
    - val_set is fixed across all folds (never in train or eval).
    - Remaining subjects are partitioned into 3 non-overlapping groups.
    - Each fold trains on 2 groups + evaluates on the 3rd.
"""

class IEMOCAP(object):
    """
    IEMOCAP Dataset Configuration

    6-modality emotion classification (4 classes).
    Modalities: video (1024), audio (1280), text (768),
                mocap_hand (18), mocap_head (6), mocap_rotated (165).
    Labels: neutral=0, angry=1, happy=2, sad=3.
    Three pre-generated 70/15/15 stratified random splits stored as CSVs
    in the data root:
        Split 0 (seed 42)  : train.csv / val.csv / test.csv
        Split 1 (seed 123) : train_split1.csv / val_split1.csv / test_split1.csv
        Split 2 (seed 456) : train_split2.csv / val_split2.csv / test_split2.csv
    """
    def __init__(self):
        super(IEMOCAP, self).__init__()
        self.num_classes = 4
        self.num_modalities = 6
        self.num_splits = 3
        self.modalities = ['video', 'audio', 'text', 'mocap_hand', 'mocap_head', 'mocap_rotated']
        self.variates = {
            'video':         1024,
            'audio':         1280,
            'text':           768,
            'mocap_hand':      18,
            'mocap_head':       6,
            'mocap_rotated':  165,
        }
        self.video_high_dim = 1024
        self.video_low_dim  = 1024



class MIMIC(object):
    def __init__(self):
        super(MIMIC, self).__init__()
        self.num_classes = 6
        self.num_modalities = 17
        self.modalities = ['glasgow','BP','HR','Temp','oxy','urine','urea','wbc','bdc2','Na','K','Bil', 'Age','icd9','hem_mal','cancer','adm_type']
        self.sampling_rates = {
            'glasgow': 1,
            'BP': 1,
            'HR': 1,
            'Temp': 1,
            'oxy': 1,
            'urine': 1,
            'urea': 1,
            'wbc': 1,
            'bdc2': 1,
            'Na' :1,
            'K' :1,
            'Bil' :1,
            'Age': 1,
            'icd9': 1,
            'hem_mal': 1,
            'cancer': 1,
            'adm_type': 1,
        }
        self.variates = {
            'glasgow': 1,
            'BP': 1,
            'HR': 1,
            'Temp': 1,
            'oxy': 1,
            'urine': 1,
            'urea': 1,
            'wbc': 1,
            'bdc2': 1,
            'Na' :1,
            'K' :1,
            'Bil' :1,
            'Age': 1,
            'icd9': 1,
            'hem_mal': 1,
            'cancer': 1,
            'adm_type': 1,
        }
        self.min_sample_rate = 1
        self.max_sample_rate = 1
        self.base_sample_rate = 1
        self.duration = 24  # 24 hours


class WESAD(object):
    """
    WESAD Dataset Configuration

    3-class stress/affect recognition from chest + wrist sensors.
    15 subjects total: S2-S11, S13-S17.
    val_set (S16, S17) is held out from all folds.
    Remaining 13 subjects split into 3 groups for CV.
    """
    def __init__(self):
        super(WESAD, self).__init__()
        self.num_classes = 3
        self.num_modalities = 10
        self.modalities = ['chest_ACC', 'chest_ECG', 'chest_EMG', 'chest_RESP', 'chest_EDA', 'chest_TEMP', 'wrist_ACC', 'wrist_BVP', 'wrist_EDA', 'wrist_TEMP']
        self.sampling_rates = {
            'chest_ACC': 700,
            'chest_ECG': 700,
            'chest_EMG': 700,
            'chest_RESP': 700,
            'chest_EDA': 700,
            'chest_TEMP': 700,
            'wrist_ACC': 32,
            'wrist_BVP': 64,
            'wrist_EDA': 4,
            'wrist_TEMP': 4,
        }
        self.variates = {
            'chest_ACC': 3,
            'chest_ECG': 1,
            'chest_EMG': 1,
            'chest_RESP': 1,
            'chest_EDA': 1,
            'chest_TEMP': 1,
            'wrist_ACC': 3,
            'wrist_BVP': 1,
            'wrist_EDA': 1,
            'wrist_TEMP': 1,
        }
        self.min_sample_rate = 4
        self.max_sample_rate = 700
        self.base_sample_rate = 32
        self.duration = 8  # seconds
        self.train_set = ['S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11']
        self.val_set = ['S16', 'S17']
        self.eval_set = ['S13', 'S14', 'S15']

        # 3-fold CV: val (S16, S17) excluded from all folds
        # Groups: A=[S2,S3,S4,S5], B=[S6,S7,S8,S9], C=[S10,S11,S13,S14,S15]
        self.folds = [
            {
                'train_set': ['S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S13', 'S14', 'S15'],
                'eval_set':  ['S2', 'S3', 'S4', 'S5'],
            },
            {
                'train_set': ['S2', 'S3', 'S4', 'S5', 'S10', 'S11', 'S13', 'S14', 'S15'],
                'eval_set':  ['S6', 'S7', 'S8', 'S9'],
            },
            {
                'train_set': ['S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9'],
                'eval_set':  ['S10', 'S11', 'S13', 'S14', 'S15'],
            },
        ]

        self.modality_scenarios = {
            
            
            ### Strong modalities 
            'chest_EDA':   ['chest_EDA'],
            'chest_ACC':   ['chest_ACC'],
            'wrist_ACC':   ['wrist_ACC'],
            'wrist_EDA':   ['wrist_EDA'],
            'C_EDA+C_ACC':     ['chest_EDA', 'chest_ACC'],
            'W_EDA+W_ACC':     ['wrist_EDA', 'wrist_ACC'],
            'C_ACC+W_ACC':     ['chest_ACC', 'wrist_ACC'],
            'C_ACC+W_ACC+W_EDA':     ['chest_ACC', 'wrist_ACC', 'wrist_EDA'],
            'C_EDA+C_ACC+W_EDA+W_ACC': ['chest_EDA', 'chest_ACC', 'wrist_EDA', 'wrist_ACC'],

            ### Weak modalities 
            'chest_ECG':   ['chest_ECG'],
            'chest_EMG':   ['chest_EMG'],
            'chest_RESP':  ['chest_RESP'],
            'chest_EMG':   ['chest_EMG'],
            'wrist_TEMP':   ['wrist_TEMP'],
            'wrist_BVP':   ['wrist_BVP'],
            'C_ECG+C_EMG':     ['chest_ECG', 'chest_EMG'],
            'C_ECG+C_EMG+C_RESP':     ['chest_ECG', 'chest_EMG', 'chest_RESP'],
            'C_ECG+C_EMG+C_RESP+W_TEMP':     ['chest_ECG', 'chest_EMG', 'chest_RESP', 'wrist_TEMP'],
            'C_ECG+C_EMG+C_RESP+W_TEMP+W_BVP':     ['chest_ECG', 'chest_EMG', 'chest_RESP', 'wrist_TEMP', 'wrist_BVP'],
            
            ### Mix modalities            
            #### All wrist
            'W_ACC+W_EDA+W_TEMP+W_BVP':     ['wrist_ACC', 'wrist_EDA', 'wrist_TEMP', 'wrist_BVP'],

            #### All chest
            'C_ECG+C_EMG+C_RESP+W_TEMP+W_BVP':     ['chest_ECG', 'chest_EMG', 'chest_RESP', 'wrist_TEMP', 'wrist_BVP'],
            'clean':       [],
        }

        # Severity ladder: 0/10 → 1/10 → 3/10 → 5/10 → 6/10 → 10/10 corrupted
        self.severity_scenarios = {
            'clean':                        [],                                                                          # 0/10
            'chest_ECG':                    ['chest_ECG'],                                                               # 1/10
            'ECG+ACC+EMG':                  ['chest_ECG', 'chest_ACC', 'chest_EMG'],                                     # 3/10
            'ECG+ACC+EMG+RESP+EDA':         ['chest_ECG', 'chest_ACC', 'chest_EMG', 'chest_RESP', 'chest_EDA'],         # 5/10
            'ECG+ACC+EMG+RESP+EDA+TEMP':    ['chest_ECG', 'chest_ACC', 'chest_EMG', 'chest_RESP', 'chest_EDA', 'chest_TEMP'],  # 6/10
            'all':                          ['chest_ECG', 'chest_ACC', 'chest_EMG', 'chest_RESP', 'chest_EDA', 'chest_TEMP', 'wrist_ACC', 'wrist_BVP', 'wrist_EDA', 'wrist_TEMP'],  # 10/10
        }


class DSADS(object):
    """
    DSADS Dataset Configuration

    Activities (19 classes):
        1. Sitting              2. Standing             3. Lying on back
        4. Lying on right side  5. Ascending stairs     6. Descending stairs
        7. Elevator (still)     8. Elevator (moving)    9. Walking (parking lot)
       10. Treadmill (flat)    11. Treadmill (incline) 12. Running (treadmill)
       13. Stepper             14. Cross trainer       15. Bike (horizontal)
       16. Bike (vertical)     17. Rowing              18. Jumping
       19. Playing basketball
    """
    def __init__(self):
        super(DSADS, self).__init__()
        self.num_classes = 19
        self.num_modalities = 5
        self.sampling_rate = 25
        self.sequence_len = 125
        self.input_channels = 45
        self.normalize = False
        self.modalities = ['torso', 'right_arm', 'left_arm', 'right_leg', 'left_leg']
        self.sampling_rates = {
            'torso': 25,
            'right_arm': 25,
            'left_arm': 25,
            'right_leg': 25,
            'left_leg': 25,
        }
        self.variates = {
            'torso': 9,
            'right_arm': 9,
            'left_arm': 9,
            'right_leg': 9,
            'left_leg': 9,
        }
        self.min_sample_rate = 25
        self.max_sample_rate = 25
        self.base_sample_rate = 25
        self.duration = 5  # seconds

        self.modality_scenarios = {
            'torso':           ['torso'],
            'right_arm':       ['right_arm'],
            'left_arm':        ['left_arm'],
            'right_leg':       ['right_leg'],
            'left_leg':        ['left_leg'],
            'arms':            ['right_arm', 'left_arm'],
            'legs':            ['right_leg', 'left_leg'],
            'torso+arms':      ['torso', 'right_arm', 'left_arm'],
            'torso+legs':      ['torso', 'right_leg', 'left_leg'],
            'all':             ['torso', 'right_arm', 'left_arm', 'right_leg', 'left_leg'],
            'clean':           [],
        }

        # Severity ladder: 0/5 → 1/5 → 2/5 → 3/5 → 5/5 corrupted
        self.severity_scenarios = {
            'clean':           [],                                                     # 0/5
            'torso':           ['torso'],                                               # 1/5
            'torso+right_arm': ['torso', 'right_arm'],                                  # 2/5
            'torso+arms':      ['torso', 'right_arm', 'left_arm'],                      # 3/5
            'all':             ['torso', 'right_arm', 'left_arm', 'right_leg', 'left_leg'],  # 5/5
        }


class DaliaHAR(object):
    """
    DaliaHAR Dataset Configuration

    7-class HAR from wrist (E4) + chest sensors.
    15 subjects: S1-S15.
    val_set (S11, S12) excluded from all folds.
    """
    def __init__(self):
        super(DaliaHAR, self).__init__()
        self.num_classes = 7
        self.num_modalities = 5
        self.modalities = ['wrist_ACC', 'wrist_BVP', 'wrist_EDA', 'wrist_TEMP', 'chest_ACC']
        self.sampling_rates = {
            'chest_ACC': 700,
            'wrist_ACC': 32,
            'wrist_BVP': 64,
            'wrist_EDA': 4,
            'wrist_TEMP': 4,
        }
        self.variates = {
            'chest_ACC': 3,
            'wrist_ACC': 3,
            'wrist_BVP': 1,
            'wrist_EDA': 1,
            'wrist_TEMP': 1,
        }
        self.min_sample_rate = 4
        self.max_sample_rate = 700
        self.base_sample_rate = 32
        self.duration = 8  # seconds
        self.train_set = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10']
        self.val_set = ['S11', 'S12']
        self.eval_set = ['S13', 'S14', 'S15']

        # 3-fold CV: val (S11, S12) excluded from all folds
        # Groups: A=[S1,S2,S3,S4], B=[S5,S6,S7,S8], C=[S9,S10,S13,S14,S15]
        self.folds = [
            {
                'train_set': ['S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S13', 'S14', 'S15'],
                'eval_set':  ['S1', 'S2', 'S3', 'S4'],
            },
            {
                'train_set': ['S1', 'S2', 'S3', 'S4', 'S9', 'S10', 'S13', 'S14', 'S15'],
                'eval_set':  ['S5', 'S6', 'S7', 'S8'],
            },
            {
                'train_set': ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8'],
                'eval_set':  ['S9', 'S10', 'S13', 'S14', 'S15'],
            },
        ]

        self.modality_scenarios = {

            ### Strong Modalities
            'wrist_ACC':     ['wrist_ACC'],
            'wrist_BVP':     ['wrist_BVP'],
            'chest_ACC':     ['chest_ACC'],
            'W_ACC+W_BVP':     ['wrist_ACC', 'wrist_BVP'],
            'C_ACC+W_ACC':     ['chest_ACC', 'wrist_ACC'],
            'W_ACC+W_BVP+C_ACC':     ['wrist_ACC', 'wrist_BVP', 'chest_ACC'],


            ### Weak Modalities
            'wrist_EDA':     ['wrist_EDA'],
            'wrist_TEMP':     ['wrist_TEMP'],
            'W_EDA+W_TEMP':     ['wrist_EDA', 'wrist_TEMP'],

            ### Mix
            'all_wrist':     ['wrist_ACC', 'wrist_BVP', 'wrist_EDA', 'wrist_TEMP'],
            'clean':         [],
        }

        # Severity ladder: 1/5 → 2/5 → 3/5 → 4/5 → 5/5 corrupted
        self.severity_scenarios = {
            'clean':                [],                                                        # 0/5
            'wrist_BVP':            ['wrist_BVP'],                                             # 1/5
            'ACC+BVP':              ['wrist_ACC', 'wrist_BVP'],                                # 2/5
            'ACC+BVP+chest':        ['wrist_ACC', 'wrist_BVP', 'chest_ACC'],                   # 3/5
            'all_wrist':            ['wrist_ACC', 'wrist_BVP', 'wrist_EDA', 'wrist_TEMP'],     # 4/5
            'all':                  ['wrist_ACC', 'wrist_BVP', 'wrist_EDA', 'wrist_TEMP', 'chest_ACC'],  # 5/5
        }


class DaliaHR(object):
    """
    DaliaHR Dataset Configuration (regression variant of DaliaHAR)

    Heart rate regression from wrist (E4) + chest sensors.
    15 subjects: S1-S15.
    val_set (S11, S12) excluded from all folds.
    """
    def __init__(self):
        super(DaliaHR, self).__init__()
        self.num_classes = 1
        self.num_modalities = 5
        self.modalities = ['wrist_ACC', 'wrist_BVP', 'wrist_EDA', 'wrist_TEMP', 'chest_ACC']
        self.sampling_rates = {
            'chest_ACC': 700,
            'wrist_ACC': 32,
            'wrist_BVP': 64,
            'wrist_EDA': 4,
            'wrist_TEMP': 4,
        }
        self.variates = {
            'chest_ACC': 3,
            'wrist_ACC': 3,
            'wrist_BVP': 1,
            'wrist_EDA': 1,
            'wrist_TEMP': 1,
        }
        self.min_sample_rate = 4
        self.max_sample_rate = 700
        self.base_sample_rate = 64
        self.duration = 8  # seconds
        self.train_set = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10']
        self.val_set = ['S11', 'S12']
        self.eval_set = ['S13', 'S14', 'S15']

        # 3-fold CV: val (S11, S12) excluded from all folds
        # Groups: A=[S1,S2,S3,S4], B=[S5,S6,S7,S8], C=[S9,S10,S13,S14,S15]
        self.folds = [
            {
                'train_set': ['S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S13', 'S14', 'S15'],
                'eval_set':  ['S1', 'S2', 'S3', 'S4'],
            },
            {
                'train_set': ['S1', 'S2', 'S3', 'S4', 'S9', 'S10', 'S13', 'S14', 'S15'],
                'eval_set':  ['S5', 'S6', 'S7', 'S8'],
            },
            {
                'train_set': ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8'],
                'eval_set':  ['S9', 'S10', 'S13', 'S14', 'S15'],
            },
        ]


class PAMAP2(object):
    """
    PAMAP2 Physical Activity Monitoring Dataset Configuration

    12-class activity recognition from 3 IMUs + heart rate.
    Pre-processed into fixed 512-length sequences.
    3 cross-validation splits (split0/split1/split2).
    """
    def __init__(self):
        super(PAMAP2, self).__init__()
        self.num_classes = 12
        self.num_modalities = 4
        self.modalities = ['IMU_hand', 'IMU_chest', 'IMU_ankle', 'heart_rate']
        self.sampling_rates = {
            'IMU_hand': 100,
            'IMU_chest': 100,
            'IMU_ankle': 100,
            'heart_rate': 100,
        }
        self.variates = {
            'IMU_hand': 17,
            'IMU_chest': 17,
            'IMU_ankle': 17,
            'heart_rate': 1,
        }
        self.min_sample_rate = 100
        self.max_sample_rate = 100
        self.base_sample_rate = 64
        self.duration = 8  # seconds -> 64 * 8 = 512 time steps
        self.sequence_length = 512
        self.num_splits = 3

        # ---- Cross-subject (subject-disjoint) protocol --------------------
        # 9 subjects S101-S109 (raw subject101..109.dat). Per-subject window
        # counts: 101:949 102:1001 103:660 104:880 105:1037 106:950 107:884
        # 108:996 109:23. S109 (tiny) always in train. Built by
        # script/pamap2_crosssubject_preprocess.py -> <cross_subject_dir>/S###.pt
        self.cross_subject_dir = '/files1/haodong/data/PAMAP2/cross_subject'
        self.subjects = ['S101', 'S102', 'S103', 'S104', 'S105',
                         'S106', 'S107', 'S108', 'S109']
        self.val_set = ['S107', 'S108']          # held out from all folds
        self.train_set = ['S101', 'S102', 'S103', 'S104', 'S105', 'S106', 'S109']
        self.eval_set = ['S107', 'S108']
        # 3-fold CV over {S101..S106} (+ S109 always train); val = S107,S108.
        self.folds = [
            {'train_set': ['S103', 'S104', 'S105', 'S106', 'S109'],
             'eval_set':  ['S101', 'S102']},
            {'train_set': ['S101', 'S102', 'S105', 'S106', 'S109'],
             'eval_set':  ['S103', 'S104']},
            {'train_set': ['S101', 'S102', 'S103', 'S104', 'S109'],
             'eval_set':  ['S105', 'S106']},
            # fold3: tests S109 (never an eval subject in folds 0-2) + S101
            {'train_set': ['S102', 'S103', 'S104', 'S105', 'S106'],
             'eval_set':  ['S101', 'S109']},
        ]

        self.modality_scenarios = {

            ### Strong Modalities
            'IMU_ankle':        ['IMU_ankle'],
            'IMU_hand':         ['IMU_hand'],
            'IMU_chest':        ['IMU_chest'],
            'ankle+hand':     ['IMU_ankle', 'IMU_hand'],
            'ankle+hand+chest':     ['IMU_ankle', 'IMU_hand', 'IMU_chest'],
            
            ### Weak Modalities
            'heart_rate':       ['heart_rate'],

            ### Mix Modalities
            'ankle+HR':       ['IMU_ankle', 'heart_rate'],
            'ankle+hand+HR':       ['IMU_ankle', 'IMU_hand', 'heart_rate'],
            'all':              ['IMU_hand', 'IMU_chest', 'IMU_ankle', 'heart_rate'],
            'clean':            [],
        }

        # Severity ladder: 0/4 → 1/4 → 2/4 → 3/4 → 4/4 corrupted
        self.severity_scenarios = {
            'clean':            [],                                                    # 0/4
            'heart_rate':       ['heart_rate'],                                        # 1/4
            'hand+HR':          ['IMU_hand', 'heart_rate'],                            # 2/4
            'hand+chest+HR':    ['IMU_hand', 'IMU_chest', 'heart_rate'],               # 3/4
            'all':              ['IMU_hand', 'IMU_chest', 'IMU_ankle', 'heart_rate'],  # 4/4
        }


class Opportunity(object):
    """
    Opportunity Activity Recognition Dataset Configuration

    Task C — Locomotion (4 classes):
        0: Stand, 1: Walk, 2: Sit, 3: Lie

    Task B2 — Mid-Level Gestures (17 classes):
        Open/Close Door1/Door2/Fridge/Dishwasher/Drawer1/Drawer2/Drawer3,
        Clean Table, Drink from Cup, Toggle Switch

    Modalities (8 body-worn sensor groups, 133 variates total):
        body_acc  : 12 body accelerometers (36-dim)
        IMU_back  : Back IMU (13-dim: acc/gyro/mag/quat)
        IMU_RUA   : Right Upper Arm IMU (13-dim)
        IMU_RLA   : Right Lower Arm IMU (13-dim)
        IMU_LUA   : Left Upper Arm IMU (13-dim)
        IMU_LLA   : Left Lower Arm IMU (13-dim)
        IMU_Lshoe : Left Shoe IMU (16-dim)
        IMU_Rshoe : Right Shoe IMU (16-dim)

    4 subjects (S1-S4), 6 sessions each (Drill + ADL1-5).
    Sampled at 30 Hz. S4 has fewer sensors (no body_acc, no shoe IMUs).
    Standard split: Train = ADL1-3 + Drill, Test = ADL4-5.
    Natural NaN sensor dropouts present (body_acc ~8%, IMUs ~1%).
    """
    def __init__(self, task='locomotion'):
        super(Opportunity, self).__init__()
        self.task = task
        self.num_classes = 4 if task == 'locomotion' else 17
        self.num_modalities = 8
        self.modalities = [
            'body_acc', 'IMU_back', 'IMU_RUA', 'IMU_RLA',
            'IMU_LUA', 'IMU_LLA', 'IMU_Lshoe', 'IMU_Rshoe',
        ]
        self.sampling_rates = {m: 30 for m in self.modalities}
        self.variates = {
            'body_acc': 36, 'IMU_back': 13, 'IMU_RUA': 13, 'IMU_RLA': 13,
            'IMU_LUA': 13, 'IMU_LLA': 13, 'IMU_Lshoe': 16, 'IMU_Rshoe': 16,
        }
        self.min_sample_rate = 30
        self.max_sample_rate = 30
        self.base_sample_rate = 30
        self.duration = 2              # seconds per window
        self.sequence_length = 60      # 30 Hz * 2s
        self.window_size = 60
        self.stride = 30               # 50% overlap

        # Column indices (0-based) in the raw .dat files
        self.sensor_col_ranges = {
            'body_acc':  (1, 37),       # cols 1-36
            'IMU_back':  (37, 50),      # cols 37-49
            'IMU_RUA':   (50, 63),      # cols 50-62
            'IMU_RLA':   (63, 76),      # cols 63-75
            'IMU_LUA':   (76, 89),      # cols 76-88
            'IMU_LLA':   (89, 102),     # cols 89-101
            'IMU_Lshoe': (102, 118),    # cols 102-117
            'IMU_Rshoe': (118, 134),    # cols 118-133
        }
        self.label_col = 243 if task == 'locomotion' else 249

        # Label mapping to contiguous 0-based indices
        if task == 'locomotion':
            self.label_map = {1: 0, 2: 1, 4: 2, 5: 3}
            self.class_names = ['Stand', 'Walk', 'Sit', 'Lie']
        else:
            self.label_map = {
                406516: 0,  404516: 1,   # Open/Close Door 1
                406517: 2,  404517: 3,   # Open/Close Door 2
                406520: 4,  404520: 5,   # Open/Close Fridge
                406505: 6,  404505: 7,   # Open/Close Dishwasher
                406519: 8,  404519: 9,   # Open/Close Drawer 1
                406511: 10, 404511: 11,  # Open/Close Drawer 2
                406508: 12, 404508: 13,  # Open/Close Drawer 3
                408512: 14, 407521: 15,  # Clean Table, Drink from Cup
                405506: 16,              # Toggle Switch
            }
            self.class_names = [
                'Open Door 1', 'Close Door 1', 'Open Door 2', 'Close Door 2',
                'Open Fridge', 'Close Fridge', 'Open Dishwasher', 'Close Dishwasher',
                'Open Drawer 1', 'Close Drawer 1', 'Open Drawer 2', 'Close Drawer 2',
                'Open Drawer 3', 'Close Drawer 3', 'Clean Table', 'Drink from Cup',
                'Toggle Switch',
            ]

        # Subjects and sessions
        self.subjects = ['S1', 'S2', 'S3']
        self.train_sessions = ['ADL1', 'ADL2', 'ADL3', 'Drill']
        self.test_sessions = ['ADL4', 'ADL5']

        # Leave-one-subject-out folds
        self.folds = [
            {'train_subjects': ['S2', 'S3'], 'eval_subjects': ['S1']},
            {'train_subjects': ['S1', 'S3'], 'eval_subjects': ['S2']},
            {'train_subjects': ['S1', 'S2'], 'eval_subjects': ['S3']},
        ]


class AngryStairs(object):
    """
    AngryOrClimbingStairs Dataset Configuration

    3-class emotion recognition (NEUTRAL, HPVHA, HNVHA) from E4 wristband.
    18 subjects: P1-P18.
    val_set (P17, P18) excluded from all folds.
    Remaining 16 subjects split into 3 groups for CV.
    """
    def __init__(self):
        super(AngryStairs, self).__init__()
        self.num_classes = 3
        self.num_modalities = 4
        self.modalities = ['e4_ACC', 'e4_BVP', 'e4_EDA', 'e4_TEMP']
        self.sampling_rates = {
            'e4_ACC': 32,
            'e4_BVP': 64,
            'e4_EDA': 4,
            'e4_TEMP': 4,
        }
        self.variates = {
            'e4_ACC': 3,
            'e4_BVP': 1,
            'e4_EDA': 1,
            'e4_TEMP': 1,
        }
        self.min_sample_rate = 4
        self.max_sample_rate = 64
        self.base_sample_rate = 32
        self.duration = 8  # seconds
        self.train_set = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10']
        self.val_set = ['P17', 'P18']
        self.eval_set = ['P11', 'P12', 'P13', 'P14', 'P15', 'P16']

        # 3-fold CV: val (P17, P18) excluded from all folds
        # Groups: A=[P1-P5], B=[P6-P10], C=[P11-P16]
        self.folds = [
            {
                'train_set': ['P6', 'P7', 'P8', 'P9', 'P10', 'P11', 'P12', 'P13', 'P14', 'P15', 'P16'],
                'eval_set':  ['P1', 'P2', 'P3', 'P4', 'P5'],
            },
            {
                'train_set': ['P1', 'P2', 'P3', 'P4', 'P5', 'P11', 'P12', 'P13', 'P14', 'P15', 'P16'],
                'eval_set':  ['P6', 'P7', 'P8', 'P9', 'P10'],
            },
            {
                'train_set': ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10'],
                'eval_set':  ['P11', 'P12', 'P13', 'P14', 'P15', 'P16'],
            },
        ]

        self.modality_scenarios = {
            'e4_ACC':       ['e4_ACC'],
            'e4_BVP':       ['e4_BVP'],
            'e4_EDA':       ['e4_EDA'],
            'e4_TEMP':      ['e4_TEMP'],
            'ACC+BVP':      ['e4_ACC', 'e4_BVP'],
            'ACC+EDA':      ['e4_ACC', 'e4_EDA'],
            'BVP+EDA':      ['e4_BVP', 'e4_EDA'],
            'ACC+BVP+EDA':  ['e4_ACC', 'e4_BVP', 'e4_EDA'],
            'all':          ['e4_ACC', 'e4_BVP', 'e4_EDA', 'e4_TEMP'],
            'clean':        [],
        }

        # Severity ladder: 0/4 → 1/4 → 2/4 → 3/4 → 4/4 corrupted
        self.severity_scenarios = {
            'clean':        [],                                          # 0/4
            'e4_BVP':       ['e4_BVP'],                                  # 1/4
            'ACC+BVP':      ['e4_ACC', 'e4_BVP'],                        # 2/4
            'ACC+BVP+EDA':  ['e4_ACC', 'e4_BVP', 'e4_EDA'],             # 3/4
            'all':          ['e4_ACC', 'e4_BVP', 'e4_EDA', 'e4_TEMP'],  # 4/4
        }


class SenseSeek(object):
    """
    SenseSeek Dataset Configuration

    6-class information-seeking behavior recognition from multimodal
    physiological signals (EDA, ACC, EEG, PUPIL, HEAD_MOTION).
    18 subjects: PA5,PA6,PA8,PA9,PA11-PA13,PA17-PA22,PA26,PA27,PA29-PA32.
    val_set (PA31, PA32) excluded from all folds.
    """
    def __init__(self):
        super(SenseSeek, self).__init__()
        self.num_classes = 6
        self.num_modalities = 5
        self.modalities = ['EDA', 'ACC', 'EEG', 'PUPIL', 'HEAD_MOTION']
        self.sampling_rates = {
            'EDA': 4,
            'ACC': 32,
            'EEG': 128,
            'PUPIL': 60,
            'HEAD_MOTION': 32,
        }
        self.variates = {
            'EDA': 3,       # Clean, Tonic, Phasic
            'ACC': 3,       # x, y, z
            'EEG': 14,      # 14 channels
            'PUPIL': 1,     # smoothed
            'HEAD_MOTION': 7,  # Q0-Q3, ACCX-Z
        }
        self.min_sample_rate = 4
        self.max_sample_rate = 128
        self.base_sample_rate = 32
        self.duration = 8  # seconds
        self.train_set = ['PA5', 'PA6', 'PA8', 'PA9', 'PA11', 'PA12', 'PA13', 'PA17', 'PA18', 'PA19']
        self.val_set = ['PA31', 'PA32']
        self.eval_set = ['PA21', 'PA22', 'PA26', 'PA27', 'PA29', 'PA30']

        # 3-fold CV: val (PA31, PA32) excluded from all folds
        # Groups: A=[PA5,PA6,PA8,PA9,PA11,PA12], B=[PA13,PA17,PA18,PA19], C=[PA21,PA22,PA26,PA27,PA29,PA30]
        self.folds = [
            {
                'train_set': ['PA13', 'PA17', 'PA18', 'PA19', 'PA21', 'PA22', 'PA26', 'PA27', 'PA29', 'PA30'],
                'eval_set':  ['PA5', 'PA6', 'PA8', 'PA9', 'PA11', 'PA12'],
            },
            {
                'train_set': ['PA5', 'PA6', 'PA8', 'PA9', 'PA11', 'PA12', 'PA21', 'PA22', 'PA26', 'PA27', 'PA29', 'PA30'],
                'eval_set':  ['PA13', 'PA17', 'PA18', 'PA19'],
            },
            {
                'train_set': ['PA5', 'PA6', 'PA8', 'PA9', 'PA11', 'PA12', 'PA13', 'PA17', 'PA18', 'PA19'],
                'eval_set':  ['PA21', 'PA22', 'PA26', 'PA27', 'PA29', 'PA30'],
            },
        ]

        self.modality_scenarios = {
            'EDA':              ['EDA'],
            'ACC':              ['ACC'],
            'EEG':              ['EEG'],
            'PUPIL':            ['PUPIL'],
            'HEAD_MOTION':      ['HEAD_MOTION'],
            'EDA+ACC':          ['EDA', 'ACC'],
            'EEG+PUPIL':        ['EEG', 'PUPIL'],
            'EDA+ACC+EEG':      ['EDA', 'ACC', 'EEG'],
            'EDA+ACC+EEG+PUPIL': ['EDA', 'ACC', 'EEG', 'PUPIL'],
            'all':              ['EDA', 'ACC', 'EEG', 'PUPIL', 'HEAD_MOTION'],
            'clean':            [],
        }

        # Severity ladder: 0/5 → 1/5 → 2/5 → 3/5 → 5/5 corrupted
        self.severity_scenarios = {
            'clean':            [],                                                     # 0/5
            'EEG':              ['EEG'],                                                # 1/5
            'EEG+ACC':          ['EEG', 'ACC'],                                         # 2/5
            'EEG+ACC+EDA':      ['EEG', 'ACC', 'EDA'],                                  # 3/5
            'all':              ['EDA', 'ACC', 'EEG', 'PUPIL', 'HEAD_MOTION'],           # 5/5
        }


class MobileCogLoad(object):
    """
    Mobile Cognitive Load Dataset Configuration

    Binary cognitive load recognition (low vs high) from MS Band 2
    wristband signals during a Snake mobile game.
    Gjoreski et al. (2020) MDPI Applied Sciences.
    Medium difficulty dropped to match paper setup (3-class was prohibitively difficult).

    Signals (~1 Hz): GSR, HR, skin temperature.
    35 usable subjects (vbd5b excluded — insufficient data).
    val_set (ya3qv, y7r8r) excluded from all folds.
    """
    def __init__(self):
        super(MobileCogLoad, self).__init__()
        self.num_classes = 2
        self.num_modalities = 3
        self.modalities = ['ms_GSR', 'ms_HR', 'ms_TEMP']
        self.sampling_rates = {
            'ms_GSR': 1,
            'ms_HR': 1,
            'ms_TEMP': 1,
        }
        self.variates = {
            'ms_GSR': 1,
            'ms_HR': 1,
            'ms_TEMP': 1,
        }
        self.min_sample_rate = 1
        self.max_sample_rate = 1
        self.base_sample_rate = 1
        self.duration = 10  # seconds (10 samples at 1 Hz)

        # 35 subjects (vbd5b excluded: only 5 windows)
        self.val_set = ['ya3qv', 'y7r8r']

        self.train_set = [
            '0dah3', '1898y', '2ceza', '2lkyl', '3190o', '5qpfr',
            '928gm', '92vgz', '93jql', 'bos1y', 'c719h', 'd1my3',
            'deztl', 'f6fbt', 'goy4y', 'hu32w', 'hznrn', 'izhf7',
            'kru08', 'kxwix', 'la3bh', 'm8rvq',
        ]
        self.eval_set = [
            'mgjvs', 'n3r2b', 'o9m76', 'p04js', 'ptoj6',
            'q0s8q', 'qwwwh', 'qy6vz', 'szs36', 'wymuo', 'xrxbc',
        ]

        # 3-fold CV: val excluded from all folds
        group_a = ['0dah3', '1898y', '2ceza', '2lkyl', '3190o', '5qpfr',
                   '928gm', '92vgz', '93jql', 'bos1y', 'c719h']
        group_b = ['d1my3', 'deztl', 'f6fbt', 'goy4y', 'hu32w', 'hznrn',
                   'izhf7', 'kru08', 'kxwix', 'la3bh', 'm8rvq']
        group_c = ['mgjvs', 'n3r2b', 'o9m76', 'p04js', 'ptoj6',
                   'q0s8q', 'qwwwh', 'qy6vz', 'szs36', 'wymuo', 'xrxbc']

        self.folds = [
            {
                'train_set': group_b + group_c,
                'eval_set':  group_a,
            },
            {
                'train_set': group_a + group_c,
                'eval_set':  group_b,
            },
            {
                'train_set': group_a + group_b,
                'eval_set':  group_c,
            },
        ]

        self.modality_scenarios = {
            'ms_GSR':       ['ms_GSR'],
            'ms_HR':        ['ms_HR'],
            'ms_TEMP':      ['ms_TEMP'],
            'GSR+HR':       ['ms_GSR', 'ms_HR'],
            'GSR+TEMP':     ['ms_GSR', 'ms_TEMP'],
            'HR+TEMP':      ['ms_HR', 'ms_TEMP'],
            'all':          ['ms_GSR', 'ms_HR', 'ms_TEMP'],
            'clean':        [],
        }

        # Severity ladder: 0/3 -> 1/3 -> 2/3 -> 3/3 corrupted
        self.severity_scenarios = {
            'clean':        [],                                  # 0/3
            'ms_GSR':       ['ms_GSR'],                          # 1/3
            'GSR+HR':       ['ms_GSR', 'ms_HR'],                 # 2/3
            'all':          ['ms_GSR', 'ms_HR', 'ms_TEMP'],      # 3/3
        }


class DEAP(object):
    """
    DEAP Dataset Configuration

    Emotion recognition from 32-channel EEG + 8 peripheral signals.
    Binary valence classification (threshold 5.0 on 1-9 scale).
    32 subjects × 40 trials × 63s (3s baseline removed → 60s stimulus).
    Preprocessed pickles: data_preprocessed_python/s{01-32}.dat
      data:   (40, 40, 8064)  — 40 trials, 40 channels, 128 Hz × 63s
      labels: (40, 4)         — valence, arousal, dominance, liking

    Channel layout:
      0-31:  EEG (32 channels)
      32:    hEOG (horizontal electrooculography)
      33:    vEOG (vertical electrooculography)
      34:    zEMG (zygomaticus EMG)
      35:    tEMG (trapezius EMG)
      36:    GSR (galvanic skin response)
      37:    Respiration
      38:    Plethysmograph (blood volume pulse)
      39:    Temperature

    9 modalities: EEG as one 32-variate modality + 8 peripheral 1-variate modalities.
    """
    def __init__(self, task='valence', threshold=5.0):
        super(DEAP, self).__init__()
        self.task = task
        self.threshold = threshold
        self.num_classes = 2
        self.num_modalities = 9
        self.baseline_samples = 3 * 128  # 3s baseline at 128 Hz
        self.modalities = [
            'EEG', 'hEOG', 'vEOG', 'zEMG', 'tEMG',
            'GSR', 'Respiration', 'Plethysmograph', 'Temperature',
        ]
        self.sampling_rates = {m: 128 for m in self.modalities}
        self.variates = {
            'EEG': 32,
            'hEOG': 1, 'vEOG': 1, 'zEMG': 1, 'tEMG': 1,
            'GSR': 1, 'Respiration': 1, 'Plethysmograph': 1, 'Temperature': 1,
        }
        self.channel_ranges = {
            'EEG': (0, 32),
            'hEOG': (32, 33), 'vEOG': (33, 34),
            'zEMG': (34, 35), 'tEMG': (35, 36),
            'GSR': (36, 37), 'Respiration': (37, 38),
            'Plethysmograph': (38, 39), 'Temperature': (39, 40),
        }
        self.label_index = {'valence': 0, 'arousal': 1, 'dominance': 2, 'liking': 3}[task]
        self.min_sample_rate = 128
        self.max_sample_rate = 128
        self.base_sample_rate = 128
        self.trial_duration = 60  # seconds (after baseline removal)
        self.window_sec = 4       # segment length in seconds
        self.stride_sec = 4       # non-overlapping
        self.duration = self.window_sec  # used by model for input_length calculation

        self.subjects = [f'S{i:02d}' for i in range(1, 33)]
        self.val_set = ['S30', 'S31', 'S32']
        self.train_set = [f'S{i:02d}' for i in range(1, 21)]
        self.eval_set = [f'S{i:02d}' for i in range(21, 30)]

        # 3-fold CV: val excluded from all folds
        group_a = [f'S{i:02d}' for i in range(1, 11)]
        group_b = [f'S{i:02d}' for i in range(11, 21)]
        group_c = [f'S{i:02d}' for i in range(21, 30)]
        self.folds = [
            {'train_set': group_b + group_c, 'eval_set': group_a},
            {'train_set': group_a + group_c, 'eval_set': group_b},
            {'train_set': group_a + group_b, 'eval_set': group_c},
        ]

        self.modality_scenarios = {
            'EEG':              ['EEG'],
            'hEOG':             ['hEOG'],
            'vEOG':             ['vEOG'],
            'zEMG':             ['zEMG'],
            'tEMG':             ['tEMG'],
            'GSR':              ['GSR'],
            'Respiration':      ['Respiration'],
            'Plethysmograph':   ['Plethysmograph'],
            'Temperature':      ['Temperature'],
            'EOG':              ['hEOG', 'vEOG'],
            'EMG':              ['zEMG', 'tEMG'],
            'EEG+EOG':          ['EEG', 'hEOG', 'vEOG'],
            'EEG+EMG':          ['EEG', 'zEMG', 'tEMG'],
            'peripheral':       ['hEOG', 'vEOG', 'zEMG', 'tEMG', 'GSR', 'Respiration', 'Plethysmograph', 'Temperature'],
            'EEG+GSR':          ['EEG', 'GSR'],
            'EEG+GSR+Resp':     ['EEG', 'GSR', 'Respiration'],
            'all':              ['EEG', 'hEOG', 'vEOG', 'zEMG', 'tEMG', 'GSR', 'Respiration', 'Plethysmograph', 'Temperature'],
            'clean':            [],
        }

        self.severity_scenarios = {
            'clean':            [],                                                                                                  # 0/9
            'Temperature':      ['Temperature'],                                                                                     # 1/9
            'Temp+Pleth':       ['Temperature', 'Plethysmograph'],                                                                   # 2/9
            'Temp+Pleth+GSR':   ['Temperature', 'Plethysmograph', 'GSR'],                                                            # 3/9
            'periph_half':      ['Temperature', 'Plethysmograph', 'GSR', 'Respiration', 'tEMG'],                                     # 5/9
            'all_peripheral':   ['hEOG', 'vEOG', 'zEMG', 'tEMG', 'GSR', 'Respiration', 'Plethysmograph', 'Temperature'],            # 8/9
            'all':              ['EEG', 'hEOG', 'vEOG', 'zEMG', 'tEMG', 'GSR', 'Respiration', 'Plethysmograph', 'Temperature'],     # 9/9
        }


class Emognition(object):
    """Emognition 2020 dataset (8 modalities, 10 emotion classes).

    Preprocessed npz at /files1/haodong/data/processed_emognition produced by
    /files1/haodong/data/Emognition/preprocess_emognition.py at 32 Hz with
    5-sec windows (160 samples) and 50% overlap.

    15 subjects with full 4-device coverage: 49-64 minus 54.
    10 classes (BASELINE excluded): AMUSEMENT, ANGER, AWE, DISGUST,
    ENTHUSIASM, FEAR, LIKING, NEUTRAL, SADNESS, SURPRISE.
    """
    def __init__(self):
        super().__init__()
        self.num_classes = 10
        self.num_modalities = 8
        self.modalities = [
            'eeg_band', 'acc_head', 'bvp', 'eda', 'skt',
            'acc_wrist', 'hr', 'face_au',
        ]
        # All modalities are resampled to 32 Hz at preprocess time.
        self.sampling_rates = {m: 32 for m in self.modalities}
        # Channel counts.
        self.variates = {
            'eeg_band': 20, 'acc_head': 3, 'bvp': 1, 'eda': 1, 'skt': 1,
            'acc_wrist': 3, 'hr': 1, 'face_au': 35,
        }
        self.min_sample_rate = 32
        self.max_sample_rate = 32
        self.base_sample_rate = 32
        self.duration = 5                                            # seconds per window
        # 3-fold LOSO over 15 subjects with rotating val (2 subj) +
        # test (5 subj) + train (8 subj). val_set is per-fold here, NOT
        # a global fixed set, because 15 subjects is too small to spare
        # a global hold-out.
        self.val_set = []                                            # unused — per-fold val
        self.folds = [
            {
                'train_set': [57, 58, 59, 60, 61, 62, 63, 64],
                'val_set':   [55, 56],
                'eval_set':  [49, 50, 51, 52, 53],
            },
            {
                'train_set': [49, 50, 51, 52, 53, 62, 63, 64],
                'val_set':   [60, 61],
                'eval_set':  [55, 56, 57, 58, 59],
            },
            {
                'train_set': [51, 52, 53, 55, 56, 57, 58, 59],
                'val_set':   [49, 50],
                'eval_set':  [60, 61, 62, 63, 64],
            },
        ]
