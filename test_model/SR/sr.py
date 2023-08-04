import replicate
output = replicate.run(
    "xpixelgroup/hat:ad47b01e4923c19fed424451d925d7e95b743be1df19e9b3b0dbb9ed8685ed6b",
    input={"image": open("/workspace/Hairthon/test_model/hairstyle_transfer/test.jpg", "rb")}
)
print(output)