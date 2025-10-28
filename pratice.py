#mark sheet
name=input("enter the name of student:")
mark=int(input("enter the mark:"))
if (mark >= 90):
    grade="A"
elif(mark >=80 and mark <=90):
    grade="B"
elif(mark >=35 and mark<=80):
    grade="C"
else:("FAIL")
print("grade of student", grade)
