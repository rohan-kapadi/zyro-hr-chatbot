import sys
import os
from pathlib import Path

# Add project directory to sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

import config
import rag_pipeline

def run_tests():
    print("=" * 60)
    print("ZYRO DYNAMICS HR HELP DESK - PIPELINE VALIDATION TEST")
    print("=" * 60)

    print("1. Configuration Check:")
    print(f"   HR Policies Directory: {config.HR_POLICIES_DIR}")
    print(f"   Exists: {config.HR_POLICIES_DIR.exists()}")
    keys_status = config.validate_api_keys()
    print(f"   Google API Key Present: {keys_status['google_api_key_configured']}")
    print(f"   LangChain Tracing: {keys_status['langsmith_tracing_enabled']}")

    print("\n2. Loading HR Policy Documents:")
    docs = rag_pipeline.load_documents()
    print(f"   Successfully loaded {len(docs)} document pages.")

    print("\n3. Splitting into Chunks:")
    chunks = rag_pipeline.split_documents(docs)
    print(f"   Created {len(chunks)} chunks with size={config.CHUNK_SIZE}, overlap={config.CHUNK_OVERLAP}.")

    print("\n4. Initializing Vector Store & Retriever:")
    vectorstore = rag_pipeline.create_vectorstore(chunks, force_rebuild=False)
    retriever = rag_pipeline.create_retriever(vectorstore)
    print("   FAISS MMR Retriever successfully initialized.")

    print("\n5. Initializing Gemini LLM & LCEL Chain:")
    llm = rag_pipeline.initialize_llm()
    chain = rag_pipeline.create_rag_chain(llm)
    print("   Gemini LLM and LCEL chain ready.")

    test_queries = [
        ("Test 1: Leave Policy", "How many annual leaves are employees entitled to?"),
        ("Test 2: Travel & Expense", "What is the reimbursement process for business travel expenses?"),
        ("Test 3: Work From Home", "What is the company's work-from-home policy?"),
        ("Test 4: Out-of-Scope", "Who won the last IPL?"),
        ("Test 5: Insufficient Info", "What is the procedure for purchasing cryptocurrency on the company credit card?")
    ]

    for label, query in test_queries:
        print("\n" + "-" * 60)
        print(f"QUERY: {label}")
        print(f"Question: \"{query}\"")
        result = rag_pipeline.ask_question(query, retriever, chain)
        print(f"Is Refusal: {result['is_refusal']}")
        print(f"Answer:\n{result['answer']}")
        print(f"Sources ({len(result['sources'])} found):")
        for idx, src in enumerate(result['sources'][:3], 1):
            print(f"   [{idx}] {src['filename']} (Page {src['page']})")
            print(f"       Excerpt: {src['excerpt'][:150]}...")

    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    run_tests()
