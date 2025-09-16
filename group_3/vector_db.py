#!/usr/bin/env python3
"""
TopDesk ChromaDB Builder for LangChain Integration
Creates a vector database from TopDesk CSV files for use with LangChain
"""

import pandas as pd
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from pathlib import Path
import logging
import json
from typing import Dict, List, Any, Optional
import re


class TopDeskVectorDB:
    """
    Builds ChromaDB vector database from TopDesk CSV files for LangChain integration
    """
    
    def __init__(self, 
                 db_path: str = "./topdesk_vectordb",
                 embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialise the vector database builder
        
        Args:
            db_path: Path to store ChromaDB database
            embedding_model: SentenceTransformer model for embeddings
        """
        self.db_path = Path(db_path)
        self.embedding_model = embedding_model
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialise ChromaDB client
        self.client = chromadb.PersistentClient(path=str(self.db_path))
        
        # Use SentenceTransformer embedding function (compatible with LangChain)
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=self.embedding_model
        )
        
        self.logger.info(f"ChromaDB initialised at: {self.db_path}")
    
    def preprocess_text(self, text: str) -> str:
        """
        Basic text preprocessing for better embeddings
        
        Args:
            text: Raw text to process
            
        Returns:
            Cleaned text
        """
        if not text or pd.isna(text):
            return ""
        
        # Convert to string and basic cleaning
        text = str(text).strip()
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text
    
    def create_incident_document(self, incident: Dict[str, Any]) -> Dict[str, str]:
        """
        Create a comprehensive document from incident data
        
        Args:
            incident: Incident data dictionary
            
        Returns:
            Document with combined text and metadata
        """
        # Combine relevant text fields
        text_parts = []
        
        if incident.get('briefDescription'):
            text_parts.append(f"Issue: {incident['briefDescription']}")
        
        if incident.get('incidentBody'):
            text_parts.append(f"Description: {incident['incidentBody']}")
        
        if incident.get('request'):
            text_parts.append(f"User Request: {incident['request']}")
        
        # Add context information
        if incident.get('category'):
            text_parts.append(f"Category: {incident['category']}")
        
        if incident.get('subcategory'):
            text_parts.append(f"Subcategory: {incident['subcategory']}")
        
        if incident.get('softwareRequired'):
            text_parts.append(f"Software: {incident['softwareRequired']}")
        
        if incident.get('callerDepartment'):
            text_parts.append(f"Department: {incident['callerDepartment']}")
        
        if incident.get('researchDiscipline'):
            text_parts.append(f"Research Area: {incident['researchDiscipline']}")
        
        # Combine all parts
        combined_text = ' | '.join(text_parts)
        cleaned_text = self.preprocess_text(combined_text)
        
        return {
            'content': cleaned_text,
            'brief_description': self.preprocess_text(incident.get('briefDescription', '')),
            'incident_body': self.preprocess_text(incident.get('incidentBody', '')),
            'category_info': f"{incident.get('category', '')} > {incident.get('subcategory', '')}"
        }
    
    def build_incidents_collection(self, incidents_csv: str) -> str:
        """
        Build incidents collection from CSV file
        
        Args:
            incidents_csv: Path to incidents CSV file
            
        Returns:
            Collection name
        """
        collection_name = "topdesk_incidents"
        
        self.logger.info(f"Building incidents collection from {incidents_csv}")
        
        # Load incidents data
        df_incidents = pd.read_csv(incidents_csv)
        self.logger.info(f"Loaded {len(df_incidents)} incidents")
        
        # Create or get collection
        try:
            self.client.delete_collection(collection_name)
        except:
            pass  # Collection doesn't exist
        
        collection = self.client.create_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
            metadata={"description": "TopDesk incidents for similarity search and categorisation"}
        )
        
        # Process incidents in batches
        documents = []
        metadatas = []
        ids = []
        
        for idx, incident in df_incidents.iterrows():
            # Create document
            doc = self.create_incident_document(incident.to_dict())
            
            if not doc['content'].strip():
                continue
            
            documents.append(doc['content'])
            
            # Create comprehensive metadata for LangChain
            metadata = {
                # Core incident info
                'id': str(incident.get('id', f'incident_{idx}')),
                'number': str(incident.get('number', '')),
                'brief_description': doc['brief_description'],
                
                # Classification
                'category': str(incident.get('category', '')),
                'subcategory': str(incident.get('subcategory', '')),
                'priority': str(incident.get('priority', '')),
                'impact': str(incident.get('impact', '')),
                'urgency': str(incident.get('urgency', '')),
                'status': str(incident.get('status', '')),
                
                # Context
                'caller_department': str(incident.get('callerDepartment', '')),
                'caller_name': str(incident.get('callerName', '')),
                'research_discipline': str(incident.get('researchDiscipline', '')),
                'software_required': str(incident.get('softwareRequired', '')),
                'object_name': str(incident.get('objectName', '')),
                'object_type': str(incident.get('objectType', '')),
                'location': str(incident.get('location', '')),
                
                # Operational
                'operator': str(incident.get('operator', '')),
                'operator_group': str(incident.get('operatorGroup', '')),
                'call_date': str(incident.get('callDate', '')),
                'creation_date': str(incident.get('creationDate', '')),
                'modification_date': str(incident.get('modificationDate', '')),
                
                # For LangChain filtering
                'document_type': 'incident',
                'source': 'topdesk'
            }
            
            metadatas.append(metadata)
            ids.append(f"incident_{incident.get('id', idx)}")
        
        # Add documents to collection in batches
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i+batch_size]
            batch_metas = metadatas[i:i+batch_size]
            batch_ids = ids[i:i+batch_size]
            
            collection.add(
                documents=batch_docs,
                metadatas=batch_metas,
                ids=batch_ids
            )
        
        self.logger.info(f"Added {len(documents)} incidents to collection '{collection_name}'")
        return collection_name
    
    def build_persons_collection(self, persons_csv: str) -> str:
        """
        Build persons collection from CSV file
        
        Args:
            persons_csv: Path to persons CSV file
            
        Returns:
            Collection name
        """
        collection_name = "topdesk_persons"
        
        self.logger.info(f"Building persons collection from {persons_csv}")
        
        # Load persons data
        df_persons = pd.read_csv(persons_csv)
        self.logger.info(f"Loaded {len(df_persons)} persons")
        
        # Create or get collection
        try:
            self.client.delete_collection(collection_name)
        except:
            pass
        
        collection = self.client.create_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
            metadata={"description": "TopDesk persons for user context and department information"}
        )
        
        # Process persons
        documents = []
        metadatas = []
        ids = []
        
        for idx, person in df_persons.iterrows():
            # Create searchable text for person
            text_parts = []
            
            if person.get('firstName') or person.get('surName'):
                name = f"{person.get('firstName', '')} {person.get('surName', '')}".strip()
                text_parts.append(f"Name: {name}")
            
            if person.get('department'):
                text_parts.append(f"Department: {person['department']}")
            
            if person.get('jobTitle'):
                text_parts.append(f"Role: {person['jobTitle']}")
            
            if person.get('branch'):
                text_parts.append(f"Location: {person['branch']}")
            
            if person.get('manager'):
                text_parts.append(f"Manager: {person['manager']}")
            
            if not text_parts:
                continue
            
            content = ' | '.join(text_parts)
            documents.append(self.preprocess_text(content))
            
            # Metadata for LangChain
            metadata = {
                'id': str(person.get('id', f'person_{idx}')),
                'dynamic_name': str(person.get('dynamicName', '')),
                'first_name': str(person.get('firstName', '')),
                'surname': str(person.get('surName', '')),
                'email': str(person.get('email', '')),
                'department': str(person.get('department', '')),
                'job_title': str(person.get('jobTitle', '')),
                'branch': str(person.get('branch', '')),
                'location': str(person.get('location', '')),
                'manager': str(person.get('manager', '')),
                'employee_number': str(person.get('employeeNumber', '')),
                'budget_holder': str(person.get('budgetHolder', '')),
                'document_type': 'person',
                'source': 'topdesk'
            }
            
            metadatas.append(metadata)
            ids.append(f"person_{person.get('id', idx)}")
        
        # Add to collection
        if documents:
            collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
        
        self.logger.info(f"Added {len(documents)} persons to collection '{collection_name}'")
        return collection_name
    
    def build_assets_collection(self, assets_csv: str) -> str:
        """
        Build assets collection from CSV file
        
        Args:
            assets_csv: Path to assets CSV file
            
        Returns:
            Collection name
        """
        collection_name = "topdesk_assets"
        
        self.logger.info(f"Building assets collection from {assets_csv}")
        
        # Load assets data
        df_assets = pd.read_csv(assets_csv)
        self.logger.info(f"Loaded {len(df_assets)} assets")
        
        # Create or get collection
        try:
            self.client.delete_collection(collection_name)
        except:
            pass
        
        collection = self.client.create_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
            metadata={"description": "TopDesk assets for hardware and equipment context"}
        )
        
        # Process assets
        documents = []
        metadatas = []
        ids = []
        
        for idx, asset in df_assets.iterrows():
            # Create searchable text for asset
            text_parts = []
            
            if asset.get('name'):
                text_parts.append(f"Asset: {asset['name']}")
            
            if asset.get('type'):
                text_parts.append(f"Type: {asset['type']}")
            
            if asset.get('brand') or asset.get('model'):
                brand_model = f"{asset.get('brand', '')} {asset.get('model', '')}".strip()
                text_parts.append(f"Device: {brand_model}")
            
            if asset.get('location'):
                text_parts.append(f"Location: {asset['location']}")
            
            if asset.get('assignedTo'):
                text_parts.append(f"Assigned to: {asset['assignedTo']}")
            
            if asset.get('assignedToDepartment'):
                text_parts.append(f"Department: {asset['assignedToDepartment']}")
            
            if not text_parts:
                continue
            
            content = ' | '.join(text_parts)
            documents.append(self.preprocess_text(content))
            
            # Metadata for LangChain
            metadata = {
                'id': str(asset.get('id', f'asset_{idx}')),
                'name': str(asset.get('name', '')),
                'type': str(asset.get('type', '')),
                'brand': str(asset.get('brand', '')),
                'model': str(asset.get('model', '')),
                'serial_number': str(asset.get('serialNumber', '')),
                'asset_tag': str(asset.get('assetTag', '')),
                'status': str(asset.get('status', '')),
                'location': str(asset.get('location', '')),
                'assigned_to': str(asset.get('assignedTo', '')),
                'assigned_to_department': str(asset.get('assignedToDepartment', '')),
                'supplier': str(asset.get('supplier', '')),
                'document_type': 'asset',
                'source': 'topdesk'
            }
            
            metadatas.append(metadata)
            ids.append(f"asset_{asset.get('id', idx)}")
        
        # Add to collection
        if documents:
            collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
        
        self.logger.info(f"Added {len(documents)} assets to collection '{collection_name}'")
        return collection_name
    
    def build_complete_database(self,
                               incidents_csv: str,
                               persons_csv: str = None,
                               assets_csv: str = None) -> Dict[str, str]:
        """
        Build complete vector database from all CSV files
        
        Args:
            incidents_csv: Path to incidents CSV
            persons_csv: Path to persons CSV (optional)
            assets_csv: Path to assets CSV (optional)
            
        Returns:
            Dictionary mapping collection names to their purposes
        """
        collections = {}
        
        # Always build incidents (main collection)
        collections['incidents'] = self.build_incidents_collection(incidents_csv)
        
        # Build optional collections
        if persons_csv:
            collections['persons'] = self.build_persons_collection(persons_csv)
        
        if assets_csv:
            collections['assets'] = self.build_assets_collection(assets_csv)
        
        # Save database info
        db_info = {
            'database_path': str(self.db_path),
            'embedding_model': self.embedding_model,
            'collections': collections,
            'total_collections': len(collections),
            'created_at': pd.Timestamp.now().isoformat()
        }
        
        # Save info file for LangChain reference
        info_path = self.db_path / 'database_info.json'
        with open(info_path, 'w') as f:
            json.dump(db_info, f, indent=2)
        
        self.logger.info("=== Database Build Complete ===")
        self.logger.info(f"Database location: {self.db_path}")
        self.logger.info(f"Collections created: {list(collections.keys())}")
        self.logger.info(f"Info saved to: {info_path}")
        self.logger.info("Ready for LangChain integration!")
        
        return collections
    
    def get_langchain_client(self):
        """
        Get ChromaDB client configured for LangChain usage
        
        Returns:
            ChromaDB client instance
        """
        return self.client
    
    def list_collections(self) -> List[str]:
        """List all available collections"""
        collections = self.client.list_collections()
        return [col.name for col in collections]
    
    def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """Get statistics for a specific collection"""
        try:
            collection = self.client.get_collection(collection_name)
            count = collection.count()
            
            # Get sample metadata to show structure
            sample = collection.peek(limit=1)
            sample_metadata = sample['metadatas'][0] if sample['metadatas'] else {}
            
            return {
                'name': collection_name,
                'document_count': count,
                'metadata_fields': list(sample_metadata.keys()) if sample_metadata else [],
                'embedding_model': self.embedding_model
            }
        except Exception as e:
            return {'error': str(e)}


def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Build TopDesk ChromaDB for LangChain')
    parser.add_argument('--incidents', required=True, help='Path to incidents CSV')
    parser.add_argument('--persons', help='Path to persons CSV')
    parser.add_argument('--assets', help='Path to assets CSV')
    parser.add_argument('--db-path', default='./topdesk_vectordb', help='Database output path')
    parser.add_argument('--embedding-model', default='all-MiniLM-L6-v2', help='Embedding model')
    
    args = parser.parse_args()
    
    # Build database
    builder = TopDeskVectorDB(args.db_path, args.embedding_model)
    collections = builder.build_complete_database(
        args.incidents,
        args.persons,
        args.assets
    )
    
    # Show collection stats
    print("\n=== Collection Statistics ===")
    for collection_name in collections.values():
        stats = builder.get_collection_stats(collection_name)
        print(f"\n{collection_name}:")
        print(f"  Documents: {stats.get('document_count', 'N/A')}")
        print(f"  Metadata fields: {len(stats.get('metadata_fields', []))}")
        print(f"  Sample fields: {stats.get('metadata_fields', [])[:5]}")


if __name__ == "__main__":
    # If no args provided, run with default files
    import sys
    if len(sys.argv) == 1:
        print("Building with default CSV files...")
        builder = TopDeskVectorDB()
        collections = builder.build_complete_database(
            'data/topdesk_incidents_dummy.csv',
            'data/topdesk_persons_dummy.csv', 
            'data/topdesk_assets_dummy.csv'
        )
    else:
        main()