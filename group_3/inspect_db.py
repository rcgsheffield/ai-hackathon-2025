#!/usr/bin/env python3
"""
ChromaDB Inspector - Check your existing ChromaDB structure
"""

import chromadb
import json
from pathlib import Path


def inspect_chromadb(db_path: str = "./topdesk_vectordb"):
    """Inspect existing ChromaDB structure"""
    
    db_path = Path(db_path)
    
    print("=== ChromaDB Structure Inspection ===\n")
    
    # Check if database exists
    if not db_path.exists():
        print(f"❌ Database path does not exist: {db_path}")
        return
    
    print(f"✓ Database path exists: {db_path}")
    
    # List contents
    contents = list(db_path.iterdir())
    print(f"✓ Database contents: {[f.name for f in contents]}")
    
    # Check database_info.json if it exists
    info_file = db_path / "database_info.json"
    if info_file.exists():
        print(f"\n=== Database Info ===")
        with open(info_file) as f:
            info = json.load(f)
        print(json.dumps(info, indent=2))
    
    # Connect to ChromaDB
    try:
        client = chromadb.PersistentClient(path=str(db_path))
        print(f"\n✓ Successfully connected to ChromaDB")
    except Exception as e:
        print(f"❌ Could not connect to ChromaDB: {e}")
        return
    
    # List collections
    try:
        collections = client.list_collections()
        print(f"\n=== Collections ({len(collections)}) ===")
        
        for col in collections:
            print(f"\nCollection: {col.name}")
            print(f"  ID: {col.id}")
            print(f"  Metadata: {col.metadata}")
            
            # Get collection details
            try:
                collection = client.get_collection(col.name)
                count = collection.count()
                print(f"  Document count: {count}")
                
                # Get a sample document to show structure
                if count > 0:
                    sample = collection.peek(limit=1)
                    if sample and sample['metadatas']:
                        sample_metadata = sample['metadatas'][0]
                        print(f"  Sample metadata keys: {list(sample_metadata.keys())}")
                        
                        # Show first few metadata values
                        print("  Sample data:")
                        for key, value in list(sample_metadata.items())[:5]:
                            print(f"    {key}: {str(value)[:50]}{'...' if len(str(value)) > 50 else ''}")
                
            except Exception as e:
                print(f"  ❌ Error accessing collection: {e}")
    
    except Exception as e:
        print(f"❌ Error listing collections: {e}")
        return
    
    print(f"\n=== Inspection Complete ===")
    
    # Recommendations
    if len(collections) == 0:
        print("⚠️  No collections found. You may need to build the vector database first.")
    elif len(collections) > 0:
        print(f"✓ Found {len(collections)} collection(s). Your ChromaDB is ready to use!")
        print(f"✓ You can now run the support ticket analyzer.")


def test_simple_query(db_path: str = "./topdesk_vectordb"):
    """Test a simple query to verify ChromaDB works"""
    
    try:
        client = chromadb.PersistentClient(path=db_path)
        collections = client.list_collections()
        
        if not collections:
            print("No collections to test")
            return
        
        # Use first collection for testing
        collection = client.get_collection(collections[0].name)
        
        print(f"\n=== Testing Query on '{collections[0].name}' ===")
        
        # Simple test query
        results = collection.query(
            query_texts=["software problem"],
            n_results=3,
            include=['documents', 'metadatas', 'distances']
        )
        
        print(f"✓ Query successful! Found {len(results['documents'][0])} results")
        
        # Show results
        for i, (doc, meta, dist) in enumerate(zip(
            results['documents'][0][:2], 
            results['metadatas'][0][:2], 
            results['distances'][0][:2]
        )):
            similarity = 1 - dist
            print(f"\nResult {i+1}:")
            print(f"  Similarity: {similarity:.1%}")
            print(f"  Category: {meta.get('category', 'N/A')}")
            print(f"  Description: {meta.get('brief_description', 'N/A')[:60]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Query test failed: {e}")
        return False


def main():
    """Main inspection function"""
    
    # Get database path
    db_path = input("Enter ChromaDB path (default: ./topdesk_vectordb): ").strip()
    if not db_path:
        db_path = "./topdesk_vectordb"
    
    # Inspect database
    inspect_chromadb(db_path)
    
    # Test query if database looks good
    collections_exist = Path(db_path).exists() and any(
        Path(db_path).iterdir()
    )
    
    if collections_exist:
        test_query = input("\nTest a simple query? (y/n): ").strip().lower()
        if test_query == 'y':
            success = test_simple_query(db_path)
            if success:
                print("\n✓ Your ChromaDB is working perfectly!")
                print("✓ You can now use the Support Ticket Analyzer")
            else:
                print("\n⚠️  ChromaDB has issues. Check the error messages above.")


if __name__ == "__main__":
    main()