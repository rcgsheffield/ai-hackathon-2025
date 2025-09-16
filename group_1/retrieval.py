from collections import defaultdict
from typing import Any, Dict, List

import pandas as pd
from langchain.chains import LLMChain
from langchain.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate


def load_faiss_vector_store(save_path: str = "faiss_index") -> FAISS:
    """
    Loads a pre-existing FAISS vector store from disk.

    Args:
        save_path (str): The directory where the FAISS index is saved.

    Returns:
        FAISS: The loaded FAISS vector store.
    """
    try:
        # The embedding model used for loading must be the same as the one used for creation
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_store = FAISS.load_local(
            save_path, embeddings, allow_dangerous_deserialization=True
        )
        print("FAISS vector store loaded successfully.")
        return vector_store
    except Exception as e:
        print(f"Error loading FAISS index: {e}")
        print(
            "Please run the indexing script first to create the faiss_index directory."
        )
        return None


def find_top_matches(
    query_text: str, vector_store: FAISS, k: int = 20
) -> List[Document]:
    """
    Performs a semantic search on the vector store to find the most relevant document chunks.

    Args:
        query_text (str): The prompt from a Group K member.
        vector_store (FAISS): The vector database to search.
        k (int): The number of top-matching chunks to retrieve.

    Returns:
        List[Document]: A list of the most relevant document chunks with their scores.
    """
    print(f"Searching for top {k} relevant chunks...")
    # Use similarity_search_with_score to get both the document and the relevance score
    retrieved_chunks = vector_store.similarity_search_with_score(query_text, k=k)
    return retrieved_chunks


def group_and_rank_profiles(retrieved_chunks: List[Any]) -> List[Dict[str, Any]]:
    """
    Groups retrieved chunks by member profile and calculates an aggregated score for each.

    Args:
        retrieved_chunks (List[Any]): A list of (Document, score) tuples.

    Returns:
        List[Dict[str, Any]]: A sorted list of dictionaries with profile details and scores.
    """
    member_scores = defaultdict(
        lambda: {"total_score": 0.0, "chunk_count": 0, "chunks": []}
    )

    for doc, score in retrieved_chunks:
        member_name = doc.metadata.get("member_name")
        if member_name:
            # We want a lower score, as it represents a smaller distance
            member_scores[member_name]["total_score"] += 1 - score
            member_scores[member_name]["chunk_count"] += 1
            member_scores[member_name]["chunks"].append(doc.page_content)

    # Convert total score to average similarity score (0 to 1)
    for member_name in member_scores:
        avg_score = (
            member_scores[member_name]["total_score"]
            / member_scores[member_name]["chunk_count"]
        )
        # Convert to percentage for display
        member_scores[member_name]["match_score_percent"] = round(avg_score * 100, 2)
        member_scores[member_name]["member_name"] = member_name

    # Sort profiles by score in descending order
    ranked_profiles = sorted(
        member_scores.values(), key=lambda x: x["match_score_percent"], reverse=True
    )

    return ranked_profiles


def generate_explanations(
    ranked_profiles: List[Dict[str, Any]], llm: ChatAnthropic
) -> None:
    """
    Generates an explanation for each top-ranked profile using an LLM.

    Args:
        ranked_profiles (List[Dict[str, Any]]): A list of the top-ranked profiles.
        llm (ChatAnthropic): The Claude LLM instance.
    """

    # Prompt template for the LLM
    prompt_template = """
    You are an expert project matcher. Your task is to analyze the provided project and interest information for two parties and generate a concise explanation of why they are a good match.
    The explanation should highlight the most significant keywords and concepts that create the match.

    Provided Context:
    - User Prompt: {user_prompt}
    - Matched Profile Content: {matched_chunks}

    Generate a brief, easy-to-read explanation (2-3 sentences max) highlighting the key points of connection.
    Example: "This profile is a strong match due to its expertise in [concept A] and [concept B], which directly relates to the user's interest in [keyword C]."
    """

    # Create a LangChain prompt template
    prompt = PromptTemplate(
        input_variables=["user_prompt", "matched_chunks"], template=prompt_template
    )

    # Use a LangChain LLMChain to combine the prompt and the LLM
    llm_chain = LLMChain(llm=llm, prompt=prompt)

    for profile in ranked_profiles:
        # Concatenate the top matching chunks to provide as context
        context_string = "\n".join(profile["chunks"])

        # Invoke the LLMChain to generate the explanation
        explanation = llm_chain.invoke(
            {
                "user_prompt": "Prompt for a Group K member (can be text from their PDFs too).",
                "matched_chunks": context_string,
            }
        )
        profile["explanation"] = explanation["text"].strip()
        print(f"Generated explanation for {profile['member_name']}.")


def display_results(ranked_profiles: List[Dict[str, Any]]):
    """
    Displays the top 10 matching profiles in a formatted table.
    """
    top_10 = ranked_profiles[:10]

    if not top_10:
        print("No matches found for the given prompt.")
        return

    # Create a pandas DataFrame for a clean table view
    df = pd.DataFrame(top_10)
    df = df[["member_name", "match_score_percent", "explanation"]]
    df.columns = ["Member", "Match Score (%)", "Explanation"]

    print("\n--- Top 10 Matching Profiles ---")
    print(df.to_string(index=False))


# --- Main Application Flow ---
if __name__ == "__main__":
    # 1. Load the FAISS vector store
    faiss_db = load_faiss_vector_store()
    if not faiss_db:
        exit()

    # 2. Define the Group K prompt
    # In a real app, this would be a user input
    user_prompt = "I'm interested in projects related to renewable energy, especially battery storage and sustainable materials for electric vehicles."

    # 3. Retrieve relevant chunks from the vector store
    retrieved_chunks = find_top_matches(user_prompt, faiss_db, k=20)

    # 4. Group chunks by profile and rank them
    ranked_profiles = group_and_rank_profiles(retrieved_chunks)

    # 5. Initialize the LLM (Claude)
    # Ensure your ANTHROPIC_API_KEY is set in your environment
    try:
        llm = ChatAnthropic(model="claude-3-haiku-20240307", temperature=0.1)
    except Exception as e:
        print(
            f"Failed to initialize Claude LLM. Ensure your API key is set. Error: {e}"
        )
        exit()

    # 6. Generate explanations for the top profiles using the LLM
    generate_explanations(ranked_profiles[:10], llm)

    # 7. Display the final results
    display_results(ranked_profiles)
