import ipywidgets
import IPython.display

# from http://chris-said.io/
toggle = '''
<script>
  function code_toggle() {
    if (code_shown){
      $('div.input').hide('100');
      $('#toggleButton').val('Show Code')
    } else {
      $('div.input').show('100');
      $('#toggleButton').val('Hide Code')
    }
    code_shown = !code_shown
  }

  $( document ).ready(function(){
    code_shown=false;
    $('div.input').hide()
  });
</script>
<form action="javascript:code_toggle()"><input type="submit" id="toggleButton" value="Show Code"></form>
'''

toggle_cell = IPython.display.HTML(toggle)

def binary_widget(default=True):
    values = [int(default), int(not default)]
    labels = [str(default), str(not default)]
    return ipywidgets.RadioWidget(values, labels=labels)
