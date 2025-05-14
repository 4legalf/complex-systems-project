from file_processing import edf_to_arr

class patient:
    def __init__(self, patient_id, name, before = None, after = None, cluster = None, response = None):
        
        self.id = patient_id
        self.name = name
        self.DMDbefore = before
        self.DMDafter = after
        self.cluster = cluster
        self.response = response
    
    def get_eeg_before(self):
        filename = f'output/raw/{self.name}S1EC-edf.csv'
        return (edf_to_arr(filename))
    
    def get_eeg_before(self):
        filename = f'output/raw/{self.name}S2EC-edf.csv'
        return (edf_to_arr(filename))