#!/bin/bash

BUCKET="gs://fdb-rag-corpus-west1/rag-corpus"
echo "Syncing rag-corpus/ to $BUCKET ..."
gsutil -m rsync -r rag-corpus/ "$BUCKET/"
echo "Done."