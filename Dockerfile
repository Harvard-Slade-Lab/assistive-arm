FROM ubuntu:20.04

# avoid getting asked questions while setting up tzdata & other packages
ARG DEBIAN_FRONTEND=noninteractive

# update the base system
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y tzdata \
    cmake \
    libpoco-dev \
    libeigen3-dev \
    mesa-utils \
    libgl1-mesa-dri \
    libosmesa6 \
    xserver-xorg-video-all \
    python3 python3-dev python3-pip \
    libtiff5-dev \
    libopenblas-dev \
    libatlas-base-dev \
    default-jdk \
    libvtk6-dev \
    libgtk-3-dev \
    libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
  
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  language-pack-en \
  bzip2 \
  sudo \
  curl \
  g++ \
  git \
  ffmpeg \
  vim \
  nano \
  wget \
  htop \
  nano \
  tmux \
  mesa-utils \
  glmark2

RUN apt update && apt install -y zsh \
  && chsh -s $(which zsh) \
  && sh -c "$(wget -O- https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" \
  && git clone https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions

# modify .zshrc to allow nice plugins
RUN sed -i '11s/ZSH_THEME="robbyrussell"/ZSH_THEME="agnoster"/' ~/.zshrc
RUN sed -i '73s/plugins=(git)/plugins=(git zsh-autosuggestions)/' ~/.zshrc

# install Python Libraries
RUN pip3 install pylint flake8

RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="${PATH}:/root/.local/bin"
ENV POETRY_VIRTUALENVS_IN_PROJECT=1

WORKDIR /home/projects/assistive-arm

CMD ["tail", "-f", "/dev/null"]