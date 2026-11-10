; Windows 11 setup for the frozen Outburst folder.
; Build with: ISCC.exe /DMyAppVersion=0.1.1 packaging\windows\Outburst.iss
; The frozen folder must already exist at dist\Outburst.

#ifndef MyAppVersion
  #define MyAppVersion "0.1.1"
#endif

#define MyAppName "Outburst"
#define MyAppExeName "Outburst.exe"

[Setup]
AppId={{7E2A6C1B-4F0D-4C8A-9B3E-1A5D8F0C2E47}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} {#MyAppVersion}
AppPublisher=Outburst contributors
AppComments=Local, offline burst culling for wildlife and nature photographers
DefaultDirName={localappdata}\Outburst
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
OutputDir=..\..\dist\installer
OutputBaseFilename=Outburst-Setup-{#MyAppVersion}
SetupIconFile=..\..\build\windows\outburst.ico
UninstallDisplayIcon={app}\{#MyAppExeName}
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0.22000
CloseApplications=yes
RestartApplications=no
UsedUserAreasWarning=no

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Add a desktop shortcut"; GroupDescription: "Shortcuts:"

[Files]
Source: "..\..\dist\Outburst\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Open Outburst"; Flags: nowait postinstall skipifsilent
