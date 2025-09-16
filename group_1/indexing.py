import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from pathlib import Path


def create_faiss_vector_store(
    acad_profiles_dir: str, save_path: str = "faiss_index"
) -> FAISS:
    """
    Processes TUOS academic profiles, creates embeddings, and stores them in a FAISS vector store.

    Args:
        acad_profiles_dir (str): The directory containing member profile folders.
        save_path (str): The directory to save the FAISS index to.

    Returns:
        FAISS: The initialized FAISS vector store.
    """
    # Initialize a list to hold all documents (text chunks) from all profiles
    all_documents = []

    # Iterate through each member's profile directory
    for member_dir in os.listdir(acad_profiles_dir):
        member_path = os.path.join(acad_profiles_dir, member_dir)
        if os.path.isdir(member_path):
            print(f"Processing profile for member: {member_dir}")

            # Combine PDF and additional text content
            full_profile_text = ""

            # Load and parse PDFs
            pdf_files = [f for f in os.listdir(member_path) if f.endswith(".pdf")]
            for pdf_file in pdf_files:
                pdf_path = os.path.join(member_path, pdf_file)
                try:
                    loader = PyPDFLoader(pdf_path)
                    pages = loader.load_and_split()
                    for page in pages:
                        full_profile_text += page.page_content + "\n"
                except Exception as e:
                    print(f"Error loading PDF {pdf_path}: {e}")

            # Load additional text from a file (e.g., a .txt file)
            text_files = [f for f in os.listdir(member_path) if f.endswith(".txt")]
            for text_file in text_files:
                text_path = os.path.join(member_path, text_file)
                try:
                    with open(text_path, "r", encoding="utf-8") as f:
                        full_profile_text += f.read() + "\n"
                except Exception as e:
                    print(f"Error loading text file {text_path}: {e}")

            # Check if there is any content to process
            if not full_profile_text.strip():
                print(f"Warning: No content found for member {member_dir}. Skipping.")
                continue

            # Create a Document object and add member metadata
            # You can also use a list of documents if you prefer
            from langchain.docstore.document import Document

            doc = Document(
                page_content=full_profile_text, metadata={"member_name": member_dir}
            )

            # Text Chunking: split the profile text into smaller, manageable chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200, length_function=len
            )
            chunks = text_splitter.split_documents([doc])
            all_documents.extend(chunks)

    # Create Embeddings
    print("Creating embeddings for all profile chunks...")
    # Use an open-source model that works well for this purpose
    # HuggingFaceEmbeddings will download the model the first time it's run
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Create and save the FAISS vector store
    print("Creating FAISS vector store and saving it locally...")
    vector_store = FAISS.from_documents(all_documents, embeddings)
    vector_store.save_local(save_path)

    print(f"FAISS vector store created and saved at: {save_path}")

    return vector_store


# Example usage:
if __name__ == "__main__":
    # Create a dummy directory structure for demonstration
    # In a real application, these profiles would already exist
    dummy_profiles_dir = "acad_profiles"
    Path(os.path.join(dummy_profiles_dir, "member1")).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(dummy_profiles_dir, "member2")).mkdir(parents=True, exist_ok=True)

    # Create dummy PDF and text files
    with open(os.path.join(dummy_profiles_dir, "member1", "project_A.pdf"), "w") as f:
        f.write(
            "This PDF discusses a project on renewable energy, specifically solar panel efficiency improvements. The project aims to reduce costs and increase power output by using novel semiconductor materials. The team has a strong interest in sustainable technologies."
        )

    with open(os.path.join(dummy_profiles_dir, "member1", "interests.txt"), "w") as f:
        f.write(
            "Additional interests include smart grid integration and energy storage solutions."
        )

    with open(os.path.join(dummy_profiles_dir, "member2", "project_B.pdf"), "w") as f:
        f.write(
            "This PDF details a project focused on developing a new type of battery for electric vehicles. The primary goal is to improve the charge density and lifespan of lithium-ion batteries. The research involves advanced chemical engineering and material science."
        )

    with open(os.path.join(dummy_profiles_dir, "member2", "bio.txt"), "w") as f:
        f.write(
            "A specialist in battery technology and electrochemical systems. Interests also include recycling and second-life applications for used batteries."
        )

    # Run the indexing function
    faiss_db = create_faiss_vector_store(acad_profiles_dir=dummy_profiles_dir)
