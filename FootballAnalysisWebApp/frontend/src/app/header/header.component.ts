import { Component, Input, OnInit } from '@angular/core';

@Component({
  selector: 'app-header',
  templateUrl: './header.component.html',
  styleUrls: ['./header.component.css']
})
export class HeaderComponent implements OnInit {

  @Input() isLogin: boolean = false; // Propriété d'entrée pour isLogin

  ngOnInit() {
  }
  deconnexion() {
    window.location.reload();
  }
}
