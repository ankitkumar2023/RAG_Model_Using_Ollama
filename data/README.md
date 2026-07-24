# data/

`vector_store/` holds the local ChromaDB persistence directory. It's
created automatically on first run and is gitignored (it's runtime
state, not source). See [docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md#known-limitation-ephemeral-storage-on-streamlit-community-cloud)
for a note on persistence when deploying to Streamlit Community Cloud.
