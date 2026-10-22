#!/bin/bash
set -e

echo "Waiting for services to start..."
echo $(pwd)

# No need to call init_nextcloud.sh and init_mattermost.sh again as they're already
# run by supervisord via their own dedicated program entries.
/init_data.sh

echo "Services are ready!"