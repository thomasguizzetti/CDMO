# Use the official MiniZinc image as the base image
FROM minizinc/minizinc

# Install Python 3, pip, Git, and other necessary packages
RUN apt-get update && apt-get install -y python3 python3-pip git

# Set the working directory in the container
WORKDIR /root

# Install any Python dependencies using pip
COPY requirements.txt /root
RUN pip3 install -r requirements.txt
RUN git clone https://github.com/thomasguizzetti/CDMO


