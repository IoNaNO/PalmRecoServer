# generate key pairs for SSL CA
openssl genrsa -out key.pem 2048
openssl req -new -key key.pem -out cert.csr
openssl x509 -req -in cert.csr -signkey key.pem -out cert.pem
rm cert.csr
# set file mode
chmod 600 key.pem
chmod 644 cert.pem