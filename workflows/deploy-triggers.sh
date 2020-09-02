BUILD_CONF='workflows/cloudbuild_template.yaml'
REPO_NAME="speech-similarity"
REPO_OWNER="vik-vok"

# cloud-func-name | py_func_name | dir
array=(
  'speech-compare-voices':'compare_voices':'functions'
)

for i in "${array[@]}"; do
  IFS=":"
  set -- ${i}

  CLOUD_FUNC_NAME=${1}
  PY_FUNC_NAME=${2}
  DIR=${3}
  TRIGGER_NAME="${CLOUD_FUNC_NAME}-trigger"
  echo "#### Generating Trigger ${TRIGGER_NAME}"

#  gcloud alpha builds triggers delete "${TRIGGER_NAME}" --quiet
  gcloud beta builds triggers create github \
    --repo-name="${REPO_NAME}" \
    --repo-owner="${REPO_OWNER}" \
    --included-files="${DIR}/*" \
    --name="${TRIGGER_NAME}" \
    --branch-pattern="^master$" \
    --build-config=${BUILD_CONF} \
    --substitutions _CLOUD_FUNC_NAME="${CLOUD_FUNC_NAME}",_PY_FUNC_NAME="${PY_FUNC_NAME}",_DIR="${DIR}"
done
