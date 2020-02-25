import Vue from 'vue'
import App from './App.vue'
import { library } from '@fortawesome/fontawesome-svg-core'
import { faRadiationAlt } from '@fortawesome/free-solid-svg-icons'
import { faGrimace } from '@fortawesome/free-regular-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/vue-fontawesome'
import { Button, Upload, Card, Dialog, Message } from 'element-ui'

Vue.config.productionTip = false

// Font Awesome
library.add(faGrimace, faRadiationAlt)
Vue.component('font-awesome-icon', FontAwesomeIcon)

// Element UI
Vue.use(Button)
Vue.use(Card)
Vue.use(Dialog)
Vue.use(Upload)

Vue.prototype.$message = Message

new Vue({
  render: h => h(App)
}).$mount('#app')
