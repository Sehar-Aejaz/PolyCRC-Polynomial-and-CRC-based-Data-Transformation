# XOR operation for two binary strings, used in CRC calculation.
def xor(a, b):
    result = []
    # Traverse each bit; XOR results in 0 if bits are same, else 1.
    for i in range(1, len(b)):
        if a[i] == b[i]:
            result.append('0')
        else:
            result.append('1')
    return ''.join(result)

# Function to perform Modulo-2 division for CRC
def mod2div(divident, divisor):
    pick = len(divisor)  # Number of bits to XOR at a time
    tmp = divident[0 : pick]  # Initial substring of divident

    while pick < len(divident):
        # Perform XOR if first bit is 1, then add next bit from divident
        if tmp[0] == '1':
            tmp = xor(divisor, tmp) + divident[pick]
        else:  # XOR with all-0 divisor if first bit is 0
            tmp = xor('0'*pick, tmp) + divident[pick]
        pick += 1

    # Handle the last step for remaining bits
    if tmp[0] == '1':
        tmp = xor(divisor, tmp)
    else:
        tmp = xor('0'*pick, tmp)

    return tmp  # Remainder of the division (CRC checkword)

# Encode data by appending CRC remainder to it
def encodeData(data, key):
    l_key = len(key)
    appended_data = data + '0'*(l_key-1)  # Append zeros for division
    remainder = mod2div(appended_data, key)  # Calculate remainder
    return data + remainder  # Return data with CRC remainder appended

# Initial number as string for conversion to binary
num = "6455919992523056"
inte = [f'{int(i):04b}' for i in str(num)]  # Convert each digit to 4-bit binary
data = ''.join(inte)  # Concatenate into a single binary string

key = "11000000000000101"  # CRC key
ans = encodeData(data, key)  # CRC encoding for the data

# Prepare original data for transformation
ini_aadhar = 6455919992523056

# Mapping digits to binary representation
con_tab = {i: f'{i:04b}' for i in range(10)}

# Converting each digit of the original number to binary
aadhar = 6455919992523056
lis = []
for i in range(17):
    if i % 4 == 0 and i > 0:
        lis.insert(0, sl)
    sl = con_tab[aadhar % 10] + sl  # Get last digit in binary
    aadhar //= 10
lis.append(ans[64:])  # Append CRC check bits to list

# Convert binary string to decimal integer
def binaryTodecimal(n):
    decimal = 0
    power = 1
    while n > 0:
        decimal += (n % 10) * power
        n //= 10
        power *= 2
    return decimal

# Calculate coefficients using binary conversion
coeff = []
for i in range(5):
    a = lis[i]
    class_ = binaryTodecimal(int(a[0] + a[8] + a[12] + a[14] + a[15]))
    row = binaryTodecimal(int(a[1:8] + a[9:12] + a[13]))
    b = (2**class_) * (2 * row + 1)  # Coefficient calculation
    coeff.append(b)

# Polynomial evaluation using Horner's method
def horner(poly, n, x):
    result = poly[0]
    for i in range(1, n):
        result = result * x + poly[i]
    return result

import pandas as pd
# Load data for vector transformation
df2 = pd.read_csv("mean_trf1.csv").drop(['Unnamed: 0'], axis=1)
vector = df2.iloc[60].to_list()

# Polynomial and its evaluation
poly = coeff[::-1]  # Polynomial coefficients in reverse
n = len(poly)
vectorproj = [horner(poly, n, i) for i in vector]

# Generate "chaff points" (random values not in original data)
import random
chaff = []
while len(chaff) < 50:
    x = round(random.uniform(-1.33, 1.66), 6)
    if x not in vector:
        chaff.append(x)

# Transform chaff points using polynomial
cpfinal = []
for x in chaff:
    cpfinal.append(horner(poly, n, x))

# Prepare data for storage and sort it
final1 = list(zip(vector, vectorproj))
final2 = list(zip(chaff, cpfinal))
final = sorted(final1 + final2)

# Select specific points for polynomial interpolation
points = [final1[i][0] for i in [0, 270, 430, 688, 1011]]
y = [final1[i][1] for i in [0, 270, 430, 688, 1011]]

# Calculate interpolation coefficients
a, b, c, d, e, f, g, h, i, j = [
    points[0] - points[1], points[0] - points[2], points[0] - points[3], points[0] - points[4],
    points[1] - points[2], points[1] - points[3], points[1] - points[4],
    points[2] - points[3], points[2] - points[4], points[3] - points[4]
]
d = [(y[0] / (a * b * c * d)), (-y[1] / (a * e * f * g)),
     (y[2] / (b * e * h * i)), (-y[3] / (c * f * h * j)),
     (y[4] / (d * g * i * j))]

# Calculate polynomial for each term
def num_poly(x):
    p = sum(x[:2] + x[2:])
    q = x[0]*x[1] + x[2]*x[3] + (x[0] + x[1]) * (x[2] + x[3])
    r = (x[0] + x[1]) * x[2] * x[3] + (x[2] + x[3]) * x[0] * x[1]
    s = x[0] * x[1] * x[2] * x[3]
    return [1, -p, q, -r, s]

polyn = []
for i in range(5):
    l = points[:i] + points[i + 1:]
    term = num_poly(l)
    polyn.append([coeff * d[i] for coeff in term])

# Sum all polynomial terms
fin = [sum(polyn[j][i] for j in range(5)) for i in range(5)]

# Final coefficients in decimal format
d_coeffs = [round(i) for i in fin[::-1]]
crc = d_coeffs.pop()
if crc == coeff[-1]:
    print("Successful Transmission")
else:
    print("Unsuccessful Transmission")

# Factorize a number into primes
def factorize(n):
    sieve = [True] * int(n ** 0.5 + 2)
    for x in range(2, int(len(sieve) ** 0.5 + 2)):
        if sieve[x]:
            for i in range(x * x, len(sieve), x):
                sieve[i] = False
    factors = [i for i in range(2, len(sieve)) if sieve[i] and n % i == 0]
    return factors if n > 1 else []

# Convert decimal coefficients back to binary for verification
d_class, d_row = [], []
for i in d_coeffs:
    fac = factorize(i)
    d_class.append(fac.count(2))
    fac = [x for x in fac if x != 2]
    prod = (prod - 1) // 2
    d_row.append(prod)

# Convert decimal to binary with zero-padding
def decimalToBinary(n):
    return bin(n).replace("0b", "")

d_lis = []
for i in range(4):
    c = decimalToBinary(d_class[i]).zfill(5)
    r = decimalToBinary(d_row[i]).zfill(11)
    d_lis.append(c[0] + r[:7] + c[1:] + r[7:] + c[3:])

# Binary to decimal for original data verification
bi = ''.join(d_lis)
d_aadhar = ''.join(str(binaryTodecimal(int(bi[i:i + 4]))) for i in range(0, 64, 4))

# Check if decrypted data matches original
if int(d_aadhar) == ini_aadhar:
    print("Successful decryption")
else:
    print("Failed decryption")
