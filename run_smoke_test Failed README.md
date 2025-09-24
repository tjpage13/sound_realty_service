Sorry you had to open this, but the .bat file failed because of computer differences in calling "python" vs "py"

In the .bat, I have:

rem 1) Create & locate venv
if not exist ".venv" (
    call :log "Creating virtual environment..."
    py -m venv .venv || (call :die "Failed to create virtual environment")
)

that needs to be replaced with:

rem 1) Create & locate venv
if not exist ".venv" (
    call :log "Creating virtual environment..."
    python -m venv .venv || (call :die "Failed to create virtual environment")
)

for your envirnoment.