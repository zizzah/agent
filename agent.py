contacts =[]

def add_contact():

 name=input("Enter name: ")
 email=input("Enter email: ")
 phone=input("Enter phone number: ")

 contact ={'name':name, 'email':email, 'phone':phone}
 contacts.append(contact)
 print("Contact added successfully!")

 for contact in contacts:
  print(f"Name: {contact['name']}, Email: {contact['email']}, Phone: {contact['phone']}")

def add(x, y):
    return x + y

def subtract(x, y):
    return x - y

def multiply(x, y):
    return x * y

def divide(x, y):
    try:
        return x / y
    except ZeroDivisionError:
        return "Error: Cannot divide by zero!"

def calculator():
    while True:
        print("\n📌 Simple Calculator")
        print("1. Add")
        print("2. Subtract")
        print("3. Multiply")
        print("4. Divide")
        print("5. Exit")

        choice = input("Choose an option (1-5): ")

        if choice == "5":
            print("Exiting Calculator... Goodbye! 👋")
            break

        if choice not in ["1", "2", "3", "4"]:
            print("Invalid choice! Please select 1-5.")
            continue

        try:
            num1 = float(input("Enter first number: "))
            num2 = float(input("Enter second number: "))
        except ValueError:
            print("Oops! Please enter valid numbers.")
            continue

        if choice == "1":
            print("Result:", add(num1, num2))
        elif choice == "2":
            print("Result:", subtract(num1, num2))
        elif choice == "3":
            print("Result:", multiply(num1, num2))
        elif choice == "4":
            print("Result:", divide(num1, num2))

# Run the calculator
calculator()
