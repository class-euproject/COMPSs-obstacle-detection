#FROM bscppc/object-trajectory-prediction-base
#ADD tracker.py /compss/
#ADD cfgfiles /compss/
#ADD stubs /compss/

FROM bscppc/object-trajectory-prediction-ubuntu-base
#ENV TZ=Europe/Madrid
ADD *.so tracker.py utils.py /compss/
ADD cfgfiles /compss/cfgfiles/
ADD stubs /compss/stubs/
#ADD phemlight /compss/phemlight/
RUN apt update && \
    apt install -y libpython3-dev libgdal-dev && \
    python3 -m pip install pygeohash && \
    apt autoremove -y && \
    rm -rf /var/lib/apt/lists