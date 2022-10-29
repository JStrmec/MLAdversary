# Dockerfile for building the ML Adversary development container
# Significant portion copied from Microsoft's base image Dockerfile
# to streamline development processes.
ARG VARIANT=latest-gpu
FROM tensorflow/tensorflow:${VARIANT}

# make library scripts directory
RUN mkdir /tmp/library-scripts

# obtain the library scripts
RUN curl https://raw.githubusercontent.com/microsoft/vscode-dev-containers/main/containers/codespaces-linux/.devcontainer/library-scripts/common-debian.sh -o /tmp/library-scripts/common-debian.sh
RUN curl https://raw.githubusercontent.com/microsoft/vscode-dev-containers/main/containers/codespaces-linux/.devcontainer/library-scripts/python-debian.sh -o /tmp/library-scripts/python-debian.sh

# [Option] Install zsh
ARG INSTALL_ZSH="true"
# [Option] Upgrade OS packages to their latest versions
ARG UPGRADE_PACKAGES="true"
# Install needed packages and setup non-root user. Use a separate RUN statement to add your own dependencies.
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    # Remove imagemagick due to https://security-tracker.debian.org/tracker/CVE-2019-10131
    && apt-get purge -y imagemagick imagemagick-6-common \
    # Install common packages, non-root user
    && bash /tmp/library-scripts/common-debian.sh "${INSTALL_ZSH}" "${USERNAME}" "${USER_UID}" "${USER_GID}" "${UPGRADE_PACKAGES}" "true" "true" \
    && apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# Remove library scripts for final image
RUN rm -rf /tmp/library-scripts

# [Optional] Allow the vscode user to pip install globally w/o sudo
ENV PIP_TARGET=/usr/local/pip-global
ENV PYTHONPATH=${PIP_TARGET}:${PYTHONPATH}
ENV PATH=${PIP_TARGET}/bin:${PATH}
RUN if ! cat /etc/group | grep -e "^pip-global:" > /dev/null 2>&1; then groupadd -r pip-global; fi \
    && usermod -a -G pip-global vscode \
    && umask 0002 && mkdir -p ${PIP_TARGET} \
    && chown :pip-global ${PIP_TARGET} \
    && ( [ ! -f "/etc/profile.d/00-restore-env.sh" ] || sed -i -e "s/export PATH=/export PATH=\/usr\/local\/pip-global:/" /etc/profile.d/00-restore-env.sh )

# grab the dataset
RUN curl https://nathanwaltz.xyz/dataset/data.zip --output data.zip
RUN unzip data.zip -d .

# create the relevant directories
RUN mkdir -p /home/ml_adversary/output
RUN mkdir -p /home/ml_adversary/saved_models
