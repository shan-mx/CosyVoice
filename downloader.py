import sys

from modelscope import snapshot_download

if len(sys.argv) < 2:
    print("Please provide the model name as a command line argument.")
    sys.exit(1)

modelName = sys.argv[1]
snapshot_download(modelName)
