import pinecone
import uuid

def save_to_pinecone(uid, embedding_list,idx_name):
    
    idx = pinecone.Index(idx_name)
    idx.upsert(items=[uid], vectors=[
            (
            f"{str(uuid.uuid4())}",                # Vector ID 
            embedding_list,  # Dense vector values
            {"genre": "cv_backend_engineer"}     # Vector metadata
            )
        ], namespace="backend-engineer-namespace")
    print("Inserted with success")