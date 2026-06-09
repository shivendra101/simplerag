As usual, you have lot of incredibly talented people offering you useful advice. You also have a few taking shots in the dark. Having done a few of these and being the middle of another implementation, here is my take :

Assuming you have already done your due-diligence as to fine-tune vs RAG, I will simply focus on RAG

Choice of VectorDB matters - for > 10 Million docs only few will stand - Weaviate, PGVector, Pinecone comes to mind. Weaviate and Pinecone have done some incredible work to optimize indexing and summarize indexing etc at that scale and that will come in handy

You need a solid Reranking strategy - RRF (Reciprocal Rank Fusion) or best yet a hybrid version of this tailored for your data set/Document content will make or break your RAG. Don't sweat too much about the embedding models - there are few good ones, choose one and focus on reranker more. You will get similar results with all of them without reranker.

Indexing - HNSW (Hierarchical Navigational Small World) Indexing strategy is a graph based multilayer indexing which is pretty solid and will give you a good balance between performance and efficacy. Make sure you choose your parameters properly _before_ you create your DB and indexing

Last but not the least - Simply throwing the documents into the ingestion pipeline will not benefit. You need a careful strategy and probably need to "segment" the documents into logical groups (Determined by your use-case/Content type) and use a "smart query router" to route it to the right Vector DB.