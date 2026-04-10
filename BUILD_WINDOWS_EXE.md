# Build Windows EXE

This project is already set up to build a standalone Windows `.exe` using PyInstaller.

## Important limitation

You must build the `.exe` on a Windows machine.

A Windows executable cannot be reliably produced from this Linux workspace with the current setup.

## Easiest path

1. Copy this project folder to a Windows computer.
2. Install Python 3.12 from `https://www.python.org/downloads/windows/`
3. During install, check `Add Python to PATH`.
4. Double-click `build.bat`
5. Wait for the build to finish.

## Output

After a successful build, give your client this folder:

`release\UtilityDashboard`

That folder will contain:

- `UtilityDashboard.exe`
- `README_FIRST.txt`

Your client only needs to double-click the `.exe`.

## If the build fails

Open Command Prompt in the project folder and run:

```bat
build.bat
```

Then read the last error shown in the terminal window.

## Recommended delivery

Send the whole `release\UtilityDashboard` folder as a `.zip` file.

Tell the client:

1. Extract the zip
2. Open the folder
3. Double-click `UtilityDashboard.exe`

