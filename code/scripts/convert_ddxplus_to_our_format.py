"""
FIXED DDXPlus to Readable Format Converter
Properly handles ALL symptom types including complex encodings
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import ast

class DDXPlusConverter:
    def __init__(self, ddxplus_path='ddxplus'):
        self.ddxplus_path = Path(ddxplus_path)
        self.symptom_mapping = {}
        self.condition_mapping = {}
        self.all_evidence_names = set()
        
    def load_metadata(self):
        """Load DDXPlus metadata files"""
        print("Loading DDXPlus metadata...")
        
        # Load release conditions (diseases)
        with open(self.ddxplus_path / 'release_conditions.json', 'r') as f:
            self.conditions = json.load(f)
        print(f"✓ Loaded {len(self.conditions)} conditions/diseases")
        
        # Load release evidences (symptoms with full info)
        with open(self.ddxplus_path / 'release_evidences.json', 'r') as f:
            self.evidences = json.load(f)
        print(f"✓ Loaded {len(self.evidences)} evidence definitions")
        
        # Create condition name mapping
        for cond_id, cond_data in self.conditions.items():
            self.condition_mapping[cond_id] = cond_data['condition_name']
        
        return self.conditions, self.evidences
    
    def build_symptom_vocabulary(self):
        """Build complete symptom vocabulary from evidences"""
        print("\nBuilding symptom vocabulary...")
    
        symptom_list = []
    
        for evidence_id, evidence_data in tqdm(self.evidences.items()):
            question = evidence_data.get('question_en', '')
            data_type = evidence_data.get('data_type', 'B')  # B=Binary, C=Categorical, M=Multi-choice
        
            # Clean the question to make it readable
            clean_question = self._clean_question(question)
        
            if data_type == 'B':
                # Binary: Just use the question as symptom name
                symptom_name = clean_question
                # Map both lowercase and original evidence_id
                self.symptom_mapping[evidence_id.lower()] = symptom_name
                self.symptom_mapping[evidence_id.upper()] = symptom_name
                symptom_list.append(symptom_name)
            
            elif data_type in ['C', 'M']:
                # Categorical/Multi-choice: Create symptom for each possible value
                possible_values = evidence_data.get('possible-values', [])
                value_meaning = evidence_data.get('value_meaning', {})
            
                for value_id in possible_values:
                    # value_id is a string like "V_161", not a dict
                    # Look it up in value_meaning to get the actual name
                    if value_id in value_meaning:
                        value_name = value_meaning[value_id].get('en', '')
                    
                        # Skip empty or "NA" values
                        if not value_name or value_name.upper() == 'NA' or value_id == 'V_11':
                            continue
                    
                        # Create compound symptom name
                        symptom_name = f"{clean_question}_{self._clean_value(value_name)}"
                    
                        # Map with @ notation (e.g., E_54_@_V_161)
                        full_key = f"{evidence_id}_@_{value_id}"
                        self.symptom_mapping[full_key.lower()] = symptom_name
                        self.symptom_mapping[full_key.upper()] = symptom_name
                    
                        symptom_list.append(symptom_name)
            
                # Also map simple evidence_id to base question
                self.symptom_mapping[evidence_id.lower()] = clean_question
                self.symptom_mapping[evidence_id.upper()] = clean_question
    
        print(f"✓ Built vocabulary with {len(set(symptom_list))} unique symptom names")
        print(f"✓ Total symptom mappings: {len(self.symptom_mapping)}")
    
        return symptom_list
    
    def _clean_value(self, value):
        """Clean value text"""
        import re
        value = value.lower()
        value = re.sub(r'[^a-z0-9\s]', '', value)
        value = value.replace(' ', '_')
        return value[:30]  # Limit length

    def _clean_question(self, question):
        """Clean question text"""
        import re
        question = question.lower()
        question = re.sub(r'[^a-z0-9\s]', '', question)
        question = question.replace(' ', '_')
        return question[:50]  # Limit length
    
    def load_patient_data(self, filename):
        """Load patient data from CSV"""
        filepath = self.ddxplus_path / filename
        print(f"\nLoading {filename}...")
        df = pd.read_csv(filepath)
        print(f"✓ Loaded {len(df):,} patients")
        return df
    
    def convert_patient_to_symptoms(self, patient_row):
        """Convert a single patient's evidences to readable symptom list"""
        symptoms = []

        # Parse the EVIDENCES column which contains a list of evidence IDs
        evidences_str = patient_row.get('EVIDENCES', '')
        if isinstance(evidences_str, str):
            try:
                evidence_list = ast.literal_eval(evidences_str)
            except (ValueError, SyntaxError):
                evidence_list = []
        else:
            evidence_list = evidences_str if isinstance(evidences_str, list) else []

        for evidence_id in evidence_list:
            # Clean the evidence_id
            evidence_id = str(evidence_id).strip()

            # Try to map the evidence
            mapped_symptom = None

            # Try exact match first
            if evidence_id in self.symptom_mapping:
                mapped_symptom = self.symptom_mapping[evidence_id]

            # Try lowercase
            elif evidence_id.lower() in self.symptom_mapping:
                mapped_symptom = self.symptom_mapping[evidence_id.lower()]

            # Try uppercase
            elif evidence_id.upper() in self.symptom_mapping:
                mapped_symptom = self.symptom_mapping[evidence_id.upper()]

            if mapped_symptom:
                symptoms.append(mapped_symptom)

        return symptoms
    
    def convert_dataset(self, train_file='release_train_patients.csv', 
                       validate_file='release_validate_patients.csv',
                       sample_size=None):
        """Convert full DDXPlus dataset to readable format"""
        
        print("\n" + "="*70)
        print("CONVERTING DDXPlus TO READABLE FORMAT")
        print("="*70)
        
        # Load metadata
        self.load_metadata()
        
        # Build symptom vocabulary
        self.build_symptom_vocabulary()
        
        # Load patient data
        train_df = self.load_patient_data(train_file)
        validate_df = self.load_patient_data(validate_file)
        
        # Combine datasets
        print("\nCombining datasets...")
        combined_df = pd.concat([train_df, validate_df], ignore_index=True)
        print(f"✓ Total patients: {len(combined_df):,}")
        
        # Sample if requested (for faster testing)
        if sample_size:
            print(f"\nSampling {sample_size:,} patients for faster processing...")
            combined_df = combined_df.sample(n=min(sample_size, len(combined_df)), random_state=42)
        
        # Convert to readable format
        print("\nConverting to readable symptom format...")
        converted_data = []

        for idx, row in tqdm(combined_df.iterrows(), total=len(combined_df)):
            # Get disease
            disease = row['PATHOLOGY']
            if disease in self.condition_mapping:
                disease = self.condition_mapping[disease]

            # Convert symptoms
            symptoms = self.convert_patient_to_symptoms(row)

            if symptoms:  # Only include if we found symptoms
                record = {'Disease': disease}
                for i, symptom in enumerate(symptoms[:20], 1):  # Max 20 symptoms
                    record[f'Symptom_{i}'] = symptom
                converted_data.append(record)

        # Create DataFrame
        if converted_data:
            result_df = pd.DataFrame(converted_data)
        else:
            # If no records were converted, create empty DataFrame with proper columns
            result_df = pd.DataFrame(columns=['Disease'] + [f'Symptom_{i}' for i in range(1, 21)])

        print(f"\n✓ Converted {len(result_df):,} records successfully")
        if not result_df.empty:
            print(f"✓ Unique diseases: {result_df['Disease'].nunique()}")

            # Get symptom columns
            symptom_cols = [col for col in result_df.columns if col.startswith('Symptom_')]
            all_symptoms = set()
            for col in symptom_cols:
                all_symptoms.update(result_df[col].dropna().unique())
            print(f"✓ Unique symptoms: {len(all_symptoms)}")
        else:
            print("✓ No records were converted - check symptom mapping")

        return result_df
    
    def create_additional_files(self, main_df):
        """Create description, precaution, and severity files"""
        
        print("\nCreating additional dataset files...")
        
        # 1. Disease Descriptions
        descriptions = []
        for disease_id, disease_data in self.conditions.items():
            disease_name = disease_data['condition_name']
            # Get symptoms associated with this disease
            symptoms = disease_data.get('symptoms', {})
            
            # Create description from available info
            if 'antecedents' in symptoms:
                desc_parts = []
                for ant in symptoms['antecedents'][:3]:  # First 3 antecedents
                    if ant in self.evidences:
                        desc_parts.append(self.evidences[ant].get('question_en', ''))
                description = f"{disease_name} is characterized by: " + "; ".join(desc_parts[:200])
            else:
                description = f"{disease_name} - a medical condition requiring professional diagnosis"
            
            descriptions.append({
                'Disease': disease_name,
                'Description': description
            })
        
        description_df = pd.DataFrame(descriptions)
        
        # 2. Precautions (generic for now)
        precautions = []
        unique_diseases = main_df['Disease'].unique()
        
        for disease in unique_diseases:
            precautions.append({
                'Disease': disease,
                'Precaution_1': 'Consult a healthcare professional immediately',
                'Precaution_2': 'Follow prescribed treatment plan',
                'Precaution_3': 'Monitor symptoms closely',
                'Precaution_4': 'Maintain good hygiene and rest'
            })
        
        precaution_df = pd.DataFrame(precautions)
        
        # 3. Symptom Severity (based on urgency if available)
        symptom_cols = [col for col in main_df.columns if col.startswith('Symptom_')]
        all_symptoms = set()
        for col in symptom_cols:
            all_symptoms.update(main_df[col].dropna().unique())
        
        severities = []
        for symptom in sorted(all_symptoms):
            # Assign severity based on keywords
            severity = 5  # Default medium
            
            symptom_lower = symptom.lower()
            if any(word in symptom_lower for word in ['severe', 'acute', 'extreme', 'high', 'intense']):
                severity = 7
            elif any(word in symptom_lower for word in ['mild', 'slight', 'minor', 'low']):
                severity = 3
            elif any(word in symptom_lower for word in ['pain', 'fever', 'bleeding', 'difficulty']):
                severity = 6
            
            severities.append({
                'Symptom': symptom,
                'weight': severity
            })
        
        severity_df = pd.DataFrame(severities)
        
        return description_df, precaution_df, severity_df
    
    def save_datasets(self, output_dir='data/raw'):
        """Convert and save all datasets"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n" + "="*70)
        print("STARTING FULL CONVERSION PROCESS")
        print("="*70)
        
        # Convert main dataset
        main_df = self.convert_dataset(sample_size=None)  # Use None for full dataset, or 10000 for testing
        
        # Save main dataset
        main_path = f"{output_dir}/dataset.csv"
        main_df.to_csv(main_path, index=False)
        print(f"\n✓ Saved main dataset: {main_path}")
        print(f"  - Total records: {len(main_df):,}")
        print(f"  - Unique diseases: {main_df['Disease'].nunique()}")
        
        # Create and save additional files
        description_df, precaution_df, severity_df = self.create_additional_files(main_df)
        
        description_df.to_csv(f"{output_dir}/symptom_Description.csv", index=False)
        print(f"✓ Saved descriptions: {output_dir}/symptom_Description.csv")
        
        precaution_df.to_csv(f"{output_dir}/symptom_precaution.csv", index=False)
        print(f"✓ Saved precautions: {output_dir}/symptom_precaution.csv")
        
        severity_df.to_csv(f"{output_dir}/Symptom-severity.csv", index=False)
        print(f"✓ Saved severities: {output_dir}/Symptom-severity.csv")
        
        # Show sample
        print("\n" + "="*70)
        print("SAMPLE DATA (with readable symptoms!):")
        print("="*70)
        print(main_df.head(3))
        
        # Verify symptoms are readable
        print("\n" + "="*70)
        print("VERIFICATION - Sample symptoms:")
        print("="*70)
        symptom_cols = [col for col in main_df.columns if col.startswith('Symptom_')]
        sample_symptoms = []
        for col in symptom_cols[:3]:
            sample_symptoms.extend(main_df[col].dropna().head(10).tolist())
        
        for i, symptom in enumerate(sample_symptoms[:20], 1):
            readable = not (symptom.startswith('E_') or '_@_' in symptom or symptom.startswith('e_'))
            status = "✓" if readable else "✗"
            print(f"  {status} {i}. {symptom}")
        
        print("\n" + "="*70)
        print("CONVERSION COMPLETE!")
        print("="*70)
        print(f"\n✓ All files saved to: {output_dir}/")
        print(f"✓ Ready to train with: python train_integrated.py")
        
        return main_df


def main():
    """Main conversion function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert DDXPlus dataset to readable format')
    parser.add_argument('--ddxplus-path', default='ddxplus', help='Path to DDXPlus directory')
    parser.add_argument('--output-dir', default='data/raw', help='Output directory')
    parser.add_argument('--sample-size', type=int, default=None, help='Sample size for testing (None for full dataset)')
    
    args = parser.parse_args()
    
    # Create converter
    converter = DDXPlusConverter(ddxplus_path=args.ddxplus_path)
    
    # Convert and save
    converter.save_datasets(output_dir=args.output_dir)
    
    print("\n" + "="*70)
    print("NEXT STEPS:")
    print("="*70)
    print("1. Verify symptoms are readable (check output above)")
    print("2. Train your model: python train_integrated.py")
    print("3. Run the app: streamlit run app_integrated.py")
    print("="*70)


if __name__ == '__main__':
    main()