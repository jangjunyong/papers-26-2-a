# d06 UIA 도우미 (PowerShell, 내장 UIAutomationClient 사용). 사용: . .\d06_uia.ps1
[Console]::OutputEncoding = [Text.Encoding]::UTF8
Add-Type -AssemblyName UIAutomationClient, UIAutomationTypes
$script:AE = [System.Windows.Automation.AutomationElement]
$script:TS = [System.Windows.Automation.TreeScope]
function Win([int]$h) { $AE::FromHandle([IntPtr]$h) }
function Kids($el) { $el.FindAll($TS::Children, [System.Windows.Automation.Condition]::TrueCondition) }
function Desc($el) { $el.FindAll($TS::Descendants, [System.Windows.Automation.Condition]::TrueCondition) }
function Dump($el, [int]$depth = 2, [int]$ind = 0) {
  foreach ($c in (Kids $el)) {
    $i = $c.Current
    $pat = ($c.GetSupportedPatterns() | ForEach-Object { $_.ProgrammaticName -replace 'Identifiers.Pattern','' }) -join ','
    ('{0}[{1}] "{2}" id={3} cls={4} rect={5},{6},{7},{8} pat={9}' -f (' ' * $ind), ($i.ControlType.ProgrammaticName -replace 'ControlType\.',''), $i.Name, $i.AutomationId, $i.ClassName, [int]$i.BoundingRectangle.X, [int]$i.BoundingRectangle.Y, [int]$i.BoundingRectangle.Width, [int]$i.BoundingRectangle.Height, $pat)
    if ($depth -gt 1) { Dump $c ($depth - 1) ($ind + 2) }
  }
}
function FindByName($el, [string]$name, $ctype = $null) {
  $cond = New-Object System.Windows.Automation.PropertyCondition($AE::NameProperty, $name)
  $el.FindFirst($TS::Descendants, $cond)
}
function InvokeEl($el) { $p = $el.GetCurrentPattern([System.Windows.Automation.InvokePattern]::Pattern); $p.Invoke() }
function FindAllType($el, $ctype) {
  $cond = New-Object System.Windows.Automation.PropertyCondition($AE::ControlTypeProperty, $ctype)
  $el.FindAll($TS::Descendants, $cond)
}
function Line($c) {
  $i = $c.Current
  $pat = ($c.GetSupportedPatterns() | ForEach-Object { $_.ProgrammaticName -replace 'Identifiers.Pattern','' }) -join ','
  '[{0}] "{1}" id={2} cls={3} rect={4},{5},{6},{7} pat={8}' -f ($i.ControlType.ProgrammaticName -replace 'ControlType\.',''), $i.Name, $i.AutomationId, $i.ClassName, [int]$i.BoundingRectangle.X, [int]$i.BoundingRectangle.Y, [int]$i.BoundingRectangle.Width, [int]$i.BoundingRectangle.Height, $pat
}
