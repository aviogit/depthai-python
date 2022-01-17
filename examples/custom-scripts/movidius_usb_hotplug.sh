#!/bin/bash

#DOMAIN='oak-d' # name of the guest VM
DOMAIN="$1" # name of the guest VM

# Abort script execution on errors
set -e
if [ "${ACTION}" == 'bind' ]; then
  COMMAND='attach-device'
elif [ "${ACTION}" == 'remove' ]; then
  COMMAND='detach-device'
  if [ "${PRODUCT}" == '3e7/2485/1' ]; then
    ID_VENDOR_ID=03e7
    ID_MODEL_ID=2485
  fi
  if [ "${PRODUCT}" == '3e7/f63b/100' ]; then
    ID_VENDOR_ID=03e7
    ID_MODEL_ID=f63b
  fi
else
  echo "Invalid udev ACTION: ${ACTION}" | logger -t 'movidius_usb_hotplug.sh'
  echo "Invalid udev ACTION: ${ACTION}" >&2
  exit 1
fi
echo "Running virsh ${COMMAND} ${DOMAIN} for ${ID_VENDOR} - vendor: 0x${ID_VENDOR_ID} - model: 0x${ID_MODEL_ID}" | logger -t 'movidius_usb_hotplug.sh'
echo "Running virsh ${COMMAND} ${DOMAIN} for ${ID_VENDOR} - vendor: 0x${ID_VENDOR_ID} - model: 0x${ID_MODEL_ID}" >&2
virsh "${COMMAND}" "${DOMAIN}" /dev/stdin <<END
<hostdev mode='subsystem' type='usb'>
  <source>
    <vendor id='0x${ID_VENDOR_ID}'/>
    <product id='0x${ID_MODEL_ID}'/>
  </source>
</hostdev>
END
exit 0
